// Function: sub_8D46E0
// Address: 0x8d46e0
//
__int64 __fastcall sub_8D46E0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  char v4; // dl
  __int64 v5; // rcx
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 result; // rax
  int v10; // r8d

  v2 = *a2;
  v3 = *a1;
  v4 = *(_BYTE *)(*a1 + 140);
  if ( v4 != 12 )
  {
LABEL_19:
    if ( *(_BYTE *)(v2 + 140) != 12 )
    {
LABEL_11:
      result = 0;
      goto LABEL_12;
    }
    goto LABEL_6;
  }
  v5 = 6338;
  while ( 1 )
  {
    v6 = *(unsigned __int8 *)(v3 + 184);
    if ( (unsigned __int8)v6 <= 0xCu )
    {
      if ( _bittest64(&v5, v6) )
        break;
    }
    v3 = *(_QWORD *)(v3 + 160);
    v4 = *(_BYTE *)(v3 + 140);
    if ( v4 != 12 )
      goto LABEL_19;
  }
  v4 = *(_BYTE *)(v2 + 140);
  if ( v4 == 12 )
  {
LABEL_6:
    v7 = 6338;
    do
    {
      v8 = *(unsigned __int8 *)(v2 + 184);
      if ( (unsigned __int8)v8 <= 0xCu && _bittest64(&v7, v8) )
      {
        if ( v4 == 12 )
          goto LABEL_15;
        goto LABEL_10;
      }
      v2 = *(_QWORD *)(v2 + 160);
    }
    while ( *(_BYTE *)(v2 + 140) == 12 );
    if ( v4 != 12 )
      goto LABEL_11;
  }
LABEL_15:
  if ( !(unsigned int)sub_8D32E0(v3) && (unsigned int)sub_8D32E0(v2) )
  {
    v2 = sub_8D46C0(v2);
    result = 1;
    goto LABEL_12;
  }
LABEL_10:
  if ( *(_BYTE *)(v2 + 140) != 12 )
    goto LABEL_11;
  v10 = sub_8D32E0(v2);
  result = 0;
  if ( !v10 )
  {
    result = sub_8D32E0(v3);
    if ( (_DWORD)result )
    {
      v3 = sub_8D46C0(v3);
      result = 1;
    }
  }
LABEL_12:
  *a1 = v3;
  *a2 = v2;
  return result;
}
