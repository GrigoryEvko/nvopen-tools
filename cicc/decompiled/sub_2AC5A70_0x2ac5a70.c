// Function: sub_2AC5A70
// Address: 0x2ac5a70
//
__int64 __fastcall sub_2AC5A70(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v4; // rax
  int v5; // ecx
  unsigned __int8 *v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  unsigned __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 *v17; // r15
  __int64 *v18; // r13
  __int64 v19; // r8
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 *v24; // rax
  __int64 *v25; // r15
  __int64 *i; // r13
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  int v35; // [rsp+24h] [rbp-7Ch]
  __int64 v36; // [rsp+30h] [rbp-70h] BYREF
  __int64 v37; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v38; // [rsp+40h] [rbp-60h] BYREF
  __int64 v39; // [rsp+48h] [rbp-58h]
  _QWORD v40[10]; // [rsp+50h] [rbp-50h] BYREF

  v4 = *(unsigned __int8 **)(a2 + 8);
  v5 = *v4;
  v40[0] = *(_QWORD *)(a3 + 8);
  v35 = v5 - 29;
  v38 = v40;
  v39 = 0x300000001LL;
  if ( (v4[7] & 0x40) != 0 )
    v6 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
  else
    v6 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  v7 = sub_2AC59D0(a1, *((_BYTE **)v6 + 4));
  v10 = HIDWORD(v39);
  v11 = v7;
  v12 = (unsigned int)v39;
  v13 = (unsigned int)v39 + 1LL;
  if ( v13 > HIDWORD(v39) )
  {
    sub_C8D5F0((__int64)&v38, v40, v13, 8u, v8, v9);
    v12 = (unsigned int)v39;
  }
  v14 = (__int64)v38;
  v38[v12] = v11;
  v15 = *(_QWORD *)(a1 + 32);
  v16 = *(_QWORD *)(a2 + 16);
  LODWORD(v39) = v39 + 1;
  if ( (unsigned __int8)sub_B19060(v15 + 440, v16, v14, v10) )
  {
    v30 = sub_2AB6F10(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 40LL));
    sub_2AB9420((__int64)&v38, v30, v31, v32, v33, v34);
  }
  v17 = v38;
  v18 = &v38[(unsigned int)v39];
  v36 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 48LL);
  if ( v36 )
    sub_2AAAFA0(&v36);
  v20 = sub_22077B0(0x68u);
  if ( v20 )
  {
    v37 = v36;
    if ( v36 )
      sub_2AAAFA0(&v37);
    *(_QWORD *)(v20 + 24) = 0;
    *(_QWORD *)(v20 + 32) = 0;
    *(_BYTE *)(v20 + 8) = 26;
    *(_QWORD *)v20 = &unk_4A231A8;
    *(_QWORD *)(v20 + 16) = 0;
    *(_QWORD *)(v20 + 48) = v20 + 64;
    *(_QWORD *)(v20 + 40) = &unk_4A23170;
    *(_QWORD *)(v20 + 56) = 0x200000000LL;
    if ( v17 != v18 )
    {
      v21 = 0;
      v22 = v20 + 64;
      v23 = *v17;
      v24 = v17 + 1;
      v25 = v18;
      for ( i = v24; ; ++i )
      {
        *(_QWORD *)(v22 + 8 * v21) = v23;
        ++*(_DWORD *)(v20 + 56);
        v27 = *(unsigned int *)(v23 + 24);
        if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 28) )
        {
          sub_C8D5F0(v23 + 16, (const void *)(v23 + 32), v27 + 1, 8u, v19, v27 + 1);
          v27 = *(unsigned int *)(v23 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v23 + 16) + 8 * v27) = v20 + 40;
        ++*(_DWORD *)(v23 + 24);
        if ( v25 == i )
          break;
        v21 = *(unsigned int *)(v20 + 56);
        v23 = *i;
        if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 60) )
        {
          sub_C8D5F0(v20 + 48, (const void *)(v20 + 64), v21 + 1, 8u, v19, v21 + 1);
          v21 = *(unsigned int *)(v20 + 56);
        }
        v22 = *(_QWORD *)(v20 + 48);
      }
    }
    *(_QWORD *)(v20 + 80) = 0;
    *(_QWORD *)(v20 + 40) = &unk_4A23AA8;
    v28 = v37;
    *(_QWORD *)v20 = &unk_4A23A70;
    *(_QWORD *)(v20 + 88) = v28;
    if ( v28 )
      sub_2AAAFA0((__int64 *)(v20 + 88));
    sub_9C6650(&v37);
    *(_QWORD *)(v20 + 40) = &unk_4A23DF8;
    *(_QWORD *)v20 = &unk_4A23DC0;
    *(_DWORD *)(v20 + 96) = v35;
  }
  sub_9C6650(&v36);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  return v20;
}
