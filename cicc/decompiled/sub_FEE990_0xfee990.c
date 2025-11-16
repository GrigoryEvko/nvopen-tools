// Function: sub_FEE990
// Address: 0xfee990
//
__int64 __fastcall sub_FEE990(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v9; // r12
  _QWORD *v10; // r15
  __int64 v11; // rax
  _QWORD v12[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v13; // [rsp+18h] [rbp-78h]
  __int64 v14; // [rsp+20h] [rbp-70h]
  void *v15; // [rsp+30h] [rbp-60h]
  _QWORD v16[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v17; // [rsp+48h] [rbp-48h]
  __int64 v18; // [rsp+50h] [rbp-40h]

  *(_QWORD *)a1 = &unk_49E55B0;
  if ( (*(_BYTE *)(a1 + 352) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 360), 24LL * *(unsigned int *)(a1 + 368), 8);
  if ( (*(_BYTE *)(a1 + 272) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 280), 16LL * *(unsigned int *)(a1 + 288), 8);
  v2 = *(_QWORD *)(a1 + 256);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 40);
    v4 = *(_QWORD *)(v2 + 32);
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(unsigned int *)(v4 + 24);
        v6 = *(_QWORD *)(v4 + 8);
        v4 += 32;
        sub_C7D6A0(v6, 16 * v5, 8);
      }
      while ( v3 != v4 );
      v4 = *(_QWORD *)(v2 + 32);
    }
    if ( v4 )
      j_j___libc_free_0(v4, *(_QWORD *)(v2 + 48) - v4);
    sub_C7D6A0(*(_QWORD *)(v2 + 8), 16LL * *(unsigned int *)(v2 + 24), 8);
    j_j___libc_free_0(v2, 56);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 216), 24LL * *(unsigned int *)(a1 + 232), 8);
  v7 = *(unsigned int *)(a1 + 200);
  if ( (_DWORD)v7 )
  {
    v9 = *(_QWORD **)(a1 + 184);
    v12[0] = 2;
    v12[1] = 0;
    v13 = -4096;
    v10 = &v9[5 * v7];
    v14 = 0;
    v16[0] = 2;
    v16[1] = 0;
    v17 = -8192;
    v15 = &unk_49DE380;
    v18 = 0;
    do
    {
      v11 = v9[3];
      *v9 = &unk_49DB368;
      if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
        sub_BD60C0(v9 + 1);
      v9 += 5;
    }
    while ( v10 != v9 );
    v15 = &unk_49DB368;
    if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
      sub_BD60C0(v16);
    if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
      sub_BD60C0(v12);
    v7 = *(unsigned int *)(a1 + 200);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 40 * v7, 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  return j_j___libc_free_0(a1, 456);
}
