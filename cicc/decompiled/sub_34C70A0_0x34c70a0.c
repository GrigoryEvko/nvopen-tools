// Function: sub_34C70A0
// Address: 0x34c70a0
//
__int64 __fastcall sub_34C70A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int8 v5; // bl
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdi
  __int64 v24; // r13
  __int64 (*v25)(); // rax
  unsigned int v26; // r13d
  __int64 v27; // rbx
  unsigned __int64 v28; // r12
  __int64 v29; // rsi
  __int64 v31; // rdx
  __int64 v32; // [rsp-10h] [rbp-170h]
  __int64 v33; // [rsp-8h] [rbp-168h]
  __int64 v34; // [rsp+0h] [rbp-160h] BYREF
  __int64 v35; // [rsp+8h] [rbp-158h]
  __int64 v36; // [rsp+10h] [rbp-150h]
  __int64 v37; // [rsp+18h] [rbp-148h]
  unsigned int v38; // [rsp+20h] [rbp-140h]
  unsigned __int64 v39; // [rsp+30h] [rbp-130h] BYREF
  __int64 v40; // [rsp+38h] [rbp-128h]
  unsigned __int64 v41; // [rsp+50h] [rbp-110h]
  char v42; // [rsp+64h] [rbp-FCh]
  __int64 v43; // [rsp+80h] [rbp-E0h]
  unsigned int v44; // [rsp+90h] [rbp-D0h]
  unsigned __int64 v45; // [rsp+98h] [rbp-C8h]
  char *v46; // [rsp+E0h] [rbp-80h]
  char v47; // [rsp+F8h] [rbp-68h] BYREF
  unsigned __int64 v48; // [rsp+108h] [rbp-58h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_48:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_48;
  }
  v5 = 0;
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_5027190);
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 8) + 688LL) & 1) == 0 )
    v5 = *(_BYTE *)(v6 + 274);
  v7 = *(__int64 **)(a1 + 8);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_45:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_501EC08 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_45;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_501EC08);
  v11 = *(__int64 **)(a1 + 8);
  v35 = 0;
  v38 = 0;
  v36 = 0;
  v37 = 0;
  v34 = v10 + 200;
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_46:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F87C64 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_46;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F87C64);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(_QWORD *)(v14 + 176);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_47:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_501F1C8 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_47;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_501F1C8);
  sub_34BEDF0((__int64)&v39, v5, 1, (__int64)&v34, v19 + 169, v16, 0);
  v20 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v23 = *(_QWORD *)(a2 + 16);
  v24 = v20;
  v25 = *(__int64 (**)())(*(_QWORD *)v23 + 128LL);
  if ( v25 == sub_2DAC790
    || (v31 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, unsigned int))v25)(
                v23,
                v5,
                v32,
                v33,
                v21,
                v22,
                v34,
                v35,
                v36,
                v37,
                v38)) == 0 )
  {
    v26 = 0;
  }
  else
  {
    v26 = sub_34C6AF0((__int64)&v39, (_QWORD *)a2, v31, v24, 0, 0);
  }
  if ( v48 )
    _libc_free(v48);
  if ( v46 != &v47 )
    _libc_free((unsigned __int64)v46);
  if ( v45 )
    j_j___libc_free_0(v45);
  sub_C7D6A0(v43, 16LL * v44, 8);
  if ( !v42 )
    _libc_free(v41);
  v27 = v40;
  v28 = v39;
  if ( v40 != v39 )
  {
    do
    {
      v29 = *(_QWORD *)(v28 + 16);
      if ( v29 )
        sub_B91220(v28 + 16, v29);
      v28 += 24LL;
    }
    while ( v27 != v28 );
    v28 = v39;
  }
  if ( v28 )
    j_j___libc_free_0(v28);
  sub_C7D6A0(v36, 16LL * v38, 8);
  return v26;
}
