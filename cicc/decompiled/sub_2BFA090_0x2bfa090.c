// Function: sub_2BFA090
// Address: 0x2bfa090
//
__int64 __fastcall sub_2BFA090(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r12
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 i; // r14
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v16[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v17; // [rsp+10h] [rbp-60h] BYREF
  void *v18; // [rsp+20h] [rbp-50h] BYREF
  __int16 v19; // [rsp+40h] [rbp-30h]

  v2 = sub_2BF9BD0(a1);
  v19 = 260;
  v18 = (void *)(a1 + 16);
  v5 = sub_22077B0(0x80u);
  if ( v5 )
  {
    sub_CA0F50(v16, &v18);
    v6 = (_BYTE *)v16[0];
    v7 = v16[1];
    *(_BYTE *)(v5 + 8) = 1;
    *(_QWORD *)v5 = &unk_4A23970;
    *(_QWORD *)(v5 + 16) = v5 + 32;
    sub_2BEF590((__int64 *)(v5 + 16), v6, (__int64)&v6[v7]);
    v8 = (__int64 *)v16[0];
    *(_QWORD *)(v5 + 56) = v5 + 72;
    *(_QWORD *)(v5 + 64) = 0x100000000LL;
    *(_QWORD *)(v5 + 88) = 0x100000000LL;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 80) = v5 + 96;
    *(_QWORD *)(v5 + 104) = 0;
    if ( v8 != &v17 )
      j_j___libc_free_0((unsigned __int64)v8);
    *(_QWORD *)v5 = &unk_4A23A00;
    *(_QWORD *)(v5 + 120) = v5 + 112;
    *(_QWORD *)(v5 + 112) = (v5 + 112) | 4;
  }
  v9 = *(unsigned int *)(v2 + 600);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 604) )
  {
    sub_C8D5F0(v2 + 592, (const void *)(v2 + 608), v9 + 1, 8u, v3, v4);
    v9 = *(unsigned int *)(v2 + 600);
  }
  v10 = a1 + 112;
  *(_QWORD *)(*(_QWORD *)(v2 + 592) + 8 * v9) = v5;
  ++*(_DWORD *)(v2 + 600);
  for ( i = *(_QWORD *)(v10 + 8); v10 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v12 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)(i - 24) + 16LL))(i - 24);
    v12[10] = v5;
    v13 = v12[3];
    v14 = *(_QWORD *)(v5 + 112);
    v12[4] = v5 + 112;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v12 + 3;
    *(_QWORD *)(v5 + 112) = *(_QWORD *)(v5 + 112) & 7LL | (unsigned __int64)(v12 + 3);
  }
  return v5;
}
