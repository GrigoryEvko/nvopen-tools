// Function: sub_1278270
// Address: 0x1278270
//
__int64 __fastcall sub_1278270(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rcx
  __int64 v5; // rax
  char v6; // si
  unsigned __int64 v7; // rdx
  bool v8; // zf
  __int64 v10; // rax
  unsigned __int64 v11; // rdx

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
LABEL_13:
    if ( v3 != *(_QWORD *)(a1 + 24) )
    {
      v10 = sub_220EF80(v3);
      v11 = *(_QWORD *)(v10 + 32);
      v3 = v10;
      v8 = v11 == *a2;
      if ( v11 >= *a2 )
        goto LABEL_15;
    }
    return 0;
  }
  v4 = *a2;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v3 + 32);
    if ( v7 > v4 || v7 == v4 && *(_BYTE *)(v3 + 40) > *((_BYTE *)a2 + 8) )
      break;
    v5 = *(_QWORD *)(v3 + 24);
    v6 = 0;
    if ( !v5 )
      goto LABEL_9;
LABEL_6:
    v3 = v5;
  }
  v5 = *(_QWORD *)(v3 + 16);
  v6 = 1;
  if ( v5 )
    goto LABEL_6;
LABEL_9:
  if ( v6 )
    goto LABEL_13;
  v8 = v7 == v4;
  if ( v7 < v4 )
    return 0;
LABEL_15:
  if ( v8 && *(_BYTE *)(v3 + 40) < *((_BYTE *)a2 + 8) )
    return 0;
  return v3;
}
