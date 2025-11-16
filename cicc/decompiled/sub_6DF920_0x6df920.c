// Function: sub_6DF920
// Address: 0x6df920
//
__int64 __fastcall sub_6DF920(__int64 *a1)
{
  __int64 v2; // rdi
  char v3; // dl
  __int64 v4; // rax
  __int64 result; // rax
  unsigned __int8 v6; // al
  _QWORD *v7; // r12
  __int64 v8; // rdx
  char i; // al

  v2 = *a1;
  v3 = *(_BYTE *)(v2 + 140);
  if ( v3 == 12 )
  {
    v4 = v2;
    do
    {
      v4 = *(_QWORD *)(v4 + 160);
      v3 = *(_BYTE *)(v4 + 140);
    }
    while ( v3 == 12 );
  }
  if ( !v3 )
    return 1;
  result = sub_8DBE70(v2);
  if ( (_DWORD)result )
  {
    if ( *((_BYTE *)a1 + 24) != 1 )
      return 1;
    v6 = *((_BYTE *)a1 + 56);
    v7 = (_QWORD *)a1[9];
    if ( v6 != 5 )
    {
      if ( v6 > 5u )
        return v6 != 116;
      if ( v6 == 3 )
        return (unsigned int)sub_8D2E30(*v7) == 0;
      return v6 != 4;
    }
    if ( (unsigned int)sub_8DD3B0(*a1) || (unsigned int)sub_8DD3B0(*v7) )
      return 1;
    v8 = *v7;
    for ( i = *(_BYTE *)(*v7 + 140LL); i == 12; i = *(_BYTE *)(v8 + 140) )
      v8 = *(_QWORD *)(v8 + 160);
    return i == 0;
  }
  return result;
}
