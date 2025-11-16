// Function: sub_1E515D0
// Address: 0x1e515d0
//
__int64 __fastcall sub_1E515D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v5; // zf
  unsigned int v9; // r9d
  int v11; // eax
  __int64 v12; // r10
  __int64 v13; // r12
  __int64 *v14; // r12
  int v15; // r9d
  __int64 v16; // rdi
  __int64 *v17; // r12
  __int64 *v18; // r10
  char v19; // al
  __int64 *v20; // [rsp+8h] [rbp-78h]
  char v21; // [rsp+8h] [rbp-78h]
  unsigned __int8 v22; // [rsp+10h] [rbp-70h]
  __int64 *v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v25[96]; // [rsp+20h] [rbp-60h] BYREF

  v5 = *(_DWORD *)(a1 + 192) == -1;
  v24 = a1;
  if ( v5 || (unsigned int)sub_1E47510(a4, a1) )
    return 0;
  v11 = sub_1E47510(a3, a1);
  v9 = 1;
  if ( v11 )
    return v9;
  sub_1E46B00((__int64)v25, a5, a1);
  if ( !v25[32] )
  {
    LOBYTE(v9) = (unsigned int)sub_1E47510(a2, v24) != 0;
    return v9;
  }
  v12 = *(_QWORD *)(v24 + 112);
  v13 = 16LL * *(unsigned int *)(v24 + 120);
  v20 = (__int64 *)(v12 + v13);
  if ( v12 != v12 + v13 )
  {
    v14 = *(__int64 **)(v24 + 112);
    LOBYTE(v15) = 0;
    do
    {
      v16 = *v14;
      v14 += 2;
      v15 = sub_1E515D0(v16 & 0xFFFFFFFFFFFFFFF8LL, a2, a3, a4, a5) | (unsigned __int8)v15;
    }
    while ( v20 != v14 );
    v17 = *(__int64 **)(v24 + 32);
    v18 = &v17[2 * *(unsigned int *)(v24 + 40)];
    if ( v18 == v17 )
      goto LABEL_14;
    goto LABEL_11;
  }
  v17 = *(__int64 **)(v24 + 32);
  LOBYTE(v15) = 0;
  v18 = &v17[2 * *(unsigned int *)(v24 + 40)];
  if ( v18 != v17 )
  {
    do
    {
LABEL_11:
      if ( ((*v17 >> 1) & 3) == 1 )
      {
        v21 = v15;
        v23 = v18;
        v19 = sub_1E515D0(*v17 & 0xFFFFFFFFFFFFFFF8LL, a2, a3, a4, a5);
        v18 = v23;
        LOBYTE(v15) = v19 | v21;
      }
      v17 += 2;
    }
    while ( v18 != v17 );
LABEL_14:
    if ( (_BYTE)v15 )
    {
      v22 = v15;
      sub_1E51470(a2, &v24);
      return v22;
    }
  }
  return 0;
}
