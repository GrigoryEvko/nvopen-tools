// Function: sub_2D1F350
// Address: 0x2d1f350
//
__int64 __fastcall sub_2D1F350(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  char v8; // r13
  __int64 v10; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 *v11; // [rsp+18h] [rbp-B8h]
  int v12; // [rsp+20h] [rbp-B0h]
  int v13; // [rsp+24h] [rbp-ACh]
  int v14; // [rsp+28h] [rbp-A8h] BYREF
  char v15; // [rsp+2Ch] [rbp-A4h]
  unsigned __int64 v16[2]; // [rsp+30h] [rbp-A0h] BYREF
  int *v17; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 *v18; // [rsp+48h] [rbp-88h]
  __int64 v19; // [rsp+50h] [rbp-80h]
  int v20; // [rsp+58h] [rbp-78h] BYREF
  char v21; // [rsp+5Ch] [rbp-74h]
  unsigned __int64 v22[5]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v23; // [rsp+88h] [rbp-48h]
  __int64 v24; // [rsp+90h] [rbp-40h]
  __int64 v25; // [rsp+98h] [rbp-38h]

  v6 = a1 + 80;
  v14 = 0;
  v10 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v11 = (unsigned __int64 *)(sub_BC1CD0(a4, &unk_4F86540, a3) + 8);
  v16[1] = (unsigned __int64)&v14;
  v17 = &v14;
  v22[1] = (unsigned __int64)&v20;
  v22[2] = (unsigned __int64)&v20;
  v16[0] = 0;
  v20 = 0;
  v18 = 0;
  v22[0] = 0;
  v22[3] = 0;
  v22[4] = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  if ( !LODWORD(qword_5016428[8]) || SLODWORD(qword_5016348[8]) <= 0 )
  {
    sub_C7D6A0(0, 0, 8);
    sub_2D1B830(v22[0]);
    sub_2D1B830(v16[0]);
    goto LABEL_8;
  }
  v8 = sub_2D1E1A0(&v10, a3);
  sub_C7D6A0(v23, 16LL * (unsigned int)v25, 8);
  sub_2D1B830(v22[0]);
  sub_2D1B830(v16[0]);
  if ( !v8 )
  {
LABEL_8:
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v6;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v12 = 2;
  v11 = v16;
  v16[0] = (unsigned __int64)&unk_4F82408;
  v14 = 0;
  v15 = 1;
  v17 = 0;
  v18 = v22;
  v19 = 2;
  v20 = 0;
  v21 = 1;
  v13 = 1;
  v10 = 1;
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v16, (__int64)&v10);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v22, (__int64)&v17);
  if ( !v21 )
  {
    _libc_free((unsigned __int64)v18);
    if ( v15 )
      return a1;
    goto LABEL_9;
  }
  if ( !v15 )
LABEL_9:
    _libc_free((unsigned __int64)v11);
  return a1;
}
