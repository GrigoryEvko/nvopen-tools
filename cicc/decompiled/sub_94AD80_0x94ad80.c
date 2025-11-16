// Function: sub_94AD80
// Address: 0x94ad80
//
__int64 __fastcall sub_94AD80(
        unsigned int **a1,
        int a2,
        int a3,
        __int64 a4,
        unsigned __int16 a5,
        int a6,
        int a7,
        unsigned __int8 a8)
{
  int v8; // r10d
  __int64 v9; // r12
  unsigned int *v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  int v19; // [rsp+4h] [rbp-7Ch]
  int v20; // [rsp+8h] [rbp-78h]
  int v21; // [rsp+8h] [rbp-78h]
  int v22; // [rsp+10h] [rbp-70h]
  unsigned __int8 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  _QWORD v26[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v27; // [rsp+40h] [rbp-40h]

  v8 = a3;
  if ( !HIBYTE(a5) )
  {
    v21 = a6;
    v25 = a4;
    v15 = sub_AA4E30(a1[6]);
    v16 = sub_9208B0(v15, *(_QWORD *)(v25 + 8));
    v26[1] = v17;
    v26[0] = (unsigned __int64)(v16 + 7) >> 3;
    v18 = sub_CA1930(v26);
    LODWORD(a4) = v25;
    v8 = a3;
    LOBYTE(a5) = -1;
    a6 = v21;
    if ( v18 )
    {
      _BitScanReverse64(&v18, v18);
      LOBYTE(a5) = 63 - (v18 ^ 0x3F);
    }
  }
  v19 = a6;
  v27 = 257;
  v20 = a4;
  v22 = v8;
  v24 = a5;
  v9 = sub_BD2C40(80, unk_3F148C4);
  if ( v9 )
    sub_B4D5A0(v9, a2, v22, v20, v24, v19, a7, a8, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    v26,
    a1[7],
    a1[8]);
  v10 = *a1;
  v11 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v11 )
  {
    do
    {
      v12 = *((_QWORD *)v10 + 1);
      v13 = *v10;
      v10 += 4;
      sub_B99FD0(v9, v13, v12);
    }
    while ( (unsigned int *)v11 != v10 );
  }
  return v9;
}
