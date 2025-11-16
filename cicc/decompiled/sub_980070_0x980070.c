// Function: sub_980070
// Address: 0x980070
//
unsigned int __fastcall sub_980070(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rax
  _BYTE v12[144]; // [rsp+0h] [rbp-100h] BYREF
  __int64 v13; // [rsp+90h] [rbp-70h]
  unsigned int v14; // [rsp+A0h] [rbp-60h]
  __int64 v15; // [rsp+B0h] [rbp-50h]
  __int64 v16; // [rsp+C0h] [rbp-40h]
  __int64 v17; // [rsp+C8h] [rbp-38h]
  __int64 v18; // [rsp+D8h] [rbp-28h]

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 4;
  *(_QWORD *)(a1 + 16) = &unk_4F6D3F0;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)a1 = &unk_49D9670;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  sub_97F3E0((__int64)v12);
  sub_97F4E0(a1 + 176, (__int64)v12);
  v2 = v17;
  *(_BYTE *)(a1 + 400) = 1;
  if ( v2 )
    j_j___libc_free_0(v2, v18 - v2);
  if ( v15 )
    j_j___libc_free_0(v15, v16 - v15);
  v3 = v14;
  if ( v14 )
  {
    v4 = v13;
    v5 = v13 + 40LL * v14;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v4 <= 0xFFFFFFFD )
        {
          v6 = *(_QWORD *)(v4 + 8);
          if ( v6 != v4 + 24 )
            break;
        }
        v4 += 40;
        if ( v5 == v4 )
          goto LABEL_11;
      }
      v7 = *(_QWORD *)(v4 + 24);
      v4 += 40;
      j_j___libc_free_0(v6, v7 + 1);
    }
    while ( v5 != v4 );
LABEL_11:
    v3 = v14;
  }
  v8 = v13;
  v9 = 40 * v3;
  sub_C7D6A0(v13, 40 * v3, 8);
  *(_BYTE *)(a1 + 488) = 0;
  v10 = sub_BC2B00(v8, v9);
  return sub_97FFF0(v10);
}
