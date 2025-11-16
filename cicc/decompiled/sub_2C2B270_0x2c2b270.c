// Function: sub_2C2B270
// Address: 0x2c2b270
//
__int64 __fastcall sub_2C2B270(__int64 a1, void **a2, _QWORD *a3)
{
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 *v12; // rdi
  __int64 v13; // rax
  __int64 v15[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF

  v9 = sub_22077B0(0x80u);
  if ( v9 )
  {
    sub_CA0F50(v15, a2);
    v10 = (_BYTE *)v15[0];
    v11 = v15[1];
    *(_BYTE *)(v9 + 8) = 1;
    *(_QWORD *)v9 = &unk_4A23970;
    *(_QWORD *)(v9 + 16) = v9 + 32;
    sub_2C256A0((__int64 *)(v9 + 16), v10, (__int64)&v10[v11]);
    v12 = (__int64 *)v15[0];
    *(_QWORD *)(v9 + 56) = v9 + 72;
    *(_QWORD *)(v9 + 64) = 0x100000000LL;
    *(_QWORD *)(v9 + 88) = 0x100000000LL;
    *(_QWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 80) = v9 + 96;
    *(_QWORD *)(v9 + 104) = 0;
    if ( v12 != &v16 )
      j_j___libc_free_0((unsigned __int64)v12);
    *(_QWORD *)v9 = &unk_4A23A00;
    *(_QWORD *)(v9 + 120) = v9 + 112;
    v5 = (v9 + 112) | 4;
    *(_QWORD *)(v9 + 112) = v5;
    if ( a3 )
    {
      a3[4] = v9 + 112;
      v13 = a3[3];
      v6 = (v9 + 112) & 0xFFFFFFFFFFFFFFF8LL;
      a3[10] = v9;
      *(_QWORD *)(v6 + 8) = a3 + 3;
      a3[3] = v6 | v13 & 7;
      v5 = *(_QWORD *)(v9 + 112) & 7LL | (unsigned __int64)(a3 + 3);
      *(_QWORD *)(v9 + 112) = v5;
    }
  }
  sub_2AB9570(a1 + 592, v9, v5, v6, v7, v8);
  return v9;
}
