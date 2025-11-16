// Function: sub_326AAE0
// Address: 0x326aae0
//
__int64 __fastcall sub_326AAE0(__int64 **a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v7; // ebx
  __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // rax
  unsigned __int16 *v12; // rax
  int v13; // r9d
  __int128 *v14; // r14
  __int64 v15; // rbx
  __int128 v16; // rax
  int v17; // r9d
  __int128 v18; // [rsp-20h] [rbp-70h]
  _BYTE v19[80]; // [rsp+0h] [rbp-50h] BYREF

  v7 = a3;
  if ( !(unsigned __int8)sub_33E03A0(**a1, a2, a3, 0) )
    return 0;
  v8 = a2[7];
  if ( !v8 )
    return 0;
  v9 = 1;
  do
  {
    while ( v7 != *(_DWORD *)(v8 + 8) )
    {
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_10;
    }
    if ( !v9 )
      return 0;
    v10 = *(_QWORD *)(v8 + 32);
    if ( !v10 )
      goto LABEL_12;
    if ( v7 == *(_DWORD *)(v10 + 8) )
      return 0;
    v8 = *(_QWORD *)(v10 + 32);
    v9 = 0;
  }
  while ( v8 );
LABEL_10:
  if ( v9 == 1 )
    return 0;
LABEL_12:
  if ( !(unsigned __int8)sub_326A930(*(_QWORD *)(a2[5] + 40LL), *(_QWORD *)(a2[5] + 48LL), 1u) )
    return 0;
  v12 = (unsigned __int16 *)(a2[6] + 16LL * v7);
  sub_2FE6CC0((__int64)v19, (*a1)[1], *(_QWORD *)(**a1 + 64), *v12, *((_QWORD *)v12 + 1));
  if ( v19[0] > 1u && !(unsigned __int8)sub_33CF530(*(_QWORD *)(a2[5] + 40LL), *(_QWORD *)(a2[5] + 48LL)) )
    return 0;
  v14 = (__int128 *)a2[5];
  v15 = **a1;
  *((_QWORD *)&v18 + 1) = a5;
  *(_QWORD *)&v18 = a4;
  *(_QWORD *)&v16 = sub_3406EB0(v15, 56, (unsigned int)a1[1], *(_DWORD *)a1[2], a1[2][1], v13, v18, *v14);
  return sub_3406EB0(
           v15,
           56,
           (unsigned int)a1[1],
           *(_DWORD *)a1[2],
           a1[2][1],
           v17,
           v16,
           *(__int128 *)((char *)v14 + 40));
}
