// Function: sub_28AF7E0
// Address: 0x28af7e0
//
__int64 __fastcall sub_28AF7E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  int v8; // edx
  __int64 v9; // rdi
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // r10
  __int64 v13; // r8
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rcx
  int v20; // eax
  int v21; // edi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 *v26; // rax
  __int64 v28; // rdi
  int v29; // ecx
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // r11
  _QWORD *v35; // rax
  __int64 v36; // r9
  _QWORD *v37; // rdi
  __int64 *v38; // rax
  __int64 v39; // rsi
  _BYTE *v40; // rax
  unsigned __int8 *v41; // rax
  char v42; // al
  int v43; // eax
  int v44; // r9d
  int v45; // eax
  int v46; // r9d
  __int64 v47; // [rsp+8h] [rbp-108h]
  __int64 v48; // [rsp+10h] [rbp-100h]
  __int64 v49; // [rsp+10h] [rbp-100h]
  __int64 v50; // [rsp+10h] [rbp-100h]
  __int64 v51; // [rsp+18h] [rbp-F8h]
  __int64 v52; // [rsp+18h] [rbp-F8h]
  _BYTE *v53; // [rsp+18h] [rbp-F8h]
  __int64 v54; // [rsp+18h] [rbp-F8h]
  _QWORD v55[6]; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v56; // [rsp+50h] [rbp-C0h] BYREF
  _QWORD v57[15]; // [rsp+60h] [rbp-B0h] BYREF

  v8 = *(_DWORD *)(a2 + 4);
  v9 = *a4;
  v56.m128i_i64[1] = 1;
  memset(v57, 0, 32);
  v55[1] = 1;
  v10 = *(_QWORD *)(a2 + 32 * (1LL - (v8 & 0x7FFFFFF)));
  memset(&v55[2], 0, 32);
  v11 = *(_DWORD *)(a3 + 4);
  v56.m128i_i64[0] = v10;
  v55[0] = *(_QWORD *)(a3 - 32LL * (v11 & 0x7FFFFFF));
  if ( (unsigned __int8)sub_CF4D50(v9, (__int64)v55, (__int64)&v56, (__int64)(a4 + 1), 0) != 3 )
    return 0;
  v12 = *(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  v13 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( v13 == v12 )
    goto LABEL_11;
  if ( *(_BYTE *)v12 != 17 || *(_BYTE *)v13 != 17 )
    return 0;
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v15 = (_QWORD *)*v15;
  if ( v15 >= v14 )
  {
    v12 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    goto LABEL_11;
  }
  v48 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v52 = *(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  sub_D671D0(&v56, a2);
  v28 = *(_QWORD *)(a1 + 40);
  v29 = *(_DWORD *)(v28 + 56);
  v30 = *(_QWORD *)(v28 + 40);
  if ( !v29 )
  {
LABEL_33:
    sub_103E0E0((_QWORD *)v28);
    BUG();
  }
  v31 = v29 - 1;
  v32 = v31 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v33 = (__int64 *)(v30 + 16LL * v32);
  v34 = *v33;
  if ( a3 != *v33 )
  {
    v45 = 1;
    while ( v34 != -4096 )
    {
      v46 = v45 + 1;
      v32 = v31 & (v45 + v32);
      v33 = (__int64 *)(v30 + 16LL * v32);
      v34 = *v33;
      if ( a3 == *v33 )
        goto LABEL_19;
      v45 = v46;
    }
    goto LABEL_33;
  }
LABEL_19:
  v47 = v48;
  v49 = v52;
  v53 = (_BYTE *)v33[1];
  v35 = sub_103E0E0((_QWORD *)v28);
  v36 = *v35;
  v37 = v35;
  v38 = (__int64 *)(v53 - 64);
  if ( *v53 == 26 )
    v38 = (__int64 *)(v53 - 32);
  v39 = *v38;
  v40 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD *, __int64, __m128i *, __int64 *))(v36 + 24))(v37, *v38, &v56, a4);
  if ( *v40 != 27 )
    return 0;
  v54 = v49;
  v50 = (__int64)v40;
  v41 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), v39);
  v42 = sub_28AAA80(*(_QWORD *)(a1 + 40), a4, v41, v50, v47);
  v12 = v54;
  if ( !v42 )
    return 0;
LABEL_11:
  v51 = v12;
  sub_23D0AB0((__int64)&v56, a2, 0, 0, 0);
  LOWORD(v16) = sub_A74840((_QWORD *)(a2 + 72), 0);
  v17 = sub_B34240(
          (__int64)&v56,
          *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
          *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))),
          v51,
          v16,
          0,
          0,
          0,
          0);
  v18 = *(_QWORD *)(a1 + 40);
  v19 = *(_QWORD *)(v18 + 40);
  v20 = *(_DWORD *)(v18 + 56);
  if ( !v20 )
  {
LABEL_24:
    v25 = 0;
    goto LABEL_14;
  }
  v21 = v20 - 1;
  v22 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v23 = (__int64 *)(v19 + 16LL * v22);
  v24 = *v23;
  if ( a2 != *v23 )
  {
    v43 = 1;
    while ( v24 != -4096 )
    {
      v44 = v43 + 1;
      v22 = v21 & (v43 + v22);
      v23 = (__int64 *)(v19 + 16LL * v22);
      v24 = *v23;
      if ( a2 == *v23 )
        goto LABEL_13;
      v43 = v44;
    }
    goto LABEL_24;
  }
LABEL_13:
  v25 = v23[1];
LABEL_14:
  v26 = (__int64 *)sub_D69570(*(_QWORD **)(a1 + 48), v17, 0, v25);
  sub_D75120(*(__int64 **)(a1 + 48), v26, 1);
  nullsub_61();
  v57[14] = &unk_49DA100;
  nullsub_63();
  if ( (_QWORD *)v56.m128i_i64[0] != v57 )
    _libc_free(v56.m128i_u64[0]);
  return 1;
}
