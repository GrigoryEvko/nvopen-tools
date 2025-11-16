// Function: sub_2362FB0
// Address: 0x2362fb0
//
__int64 *__fastcall sub_2362FB0(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 **v8; // r8
  unsigned int v9; // edx
  _QWORD *v10; // rbx
  __int64 *v11; // r9
  __int64 *v12; // rbx
  __int64 *v13; // rax
  unsigned int v14; // esi
  __int64 v15; // rcx
  int v16; // r10d
  __int64 **v17; // r8
  unsigned int v18; // edx
  _QWORD *v19; // rbx
  __int64 *v20; // r9
  __int64 *v21; // rbx
  __int64 *result; // rax
  __int64 **v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rcx
  int v26; // r10d
  __int64 **v27; // r8
  unsigned __int64 v28; // rdx
  __int64 **v29; // rbx
  __int64 *v30; // r9
  __int64 **v31; // rbx
  __int64 v32; // rbx
  __int64 i; // r13
  _QWORD *v34; // rax
  __int64 v35; // rdi
  _QWORD *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // r14
  int v39; // ecx
  int v40; // ecx
  int v41; // ecx
  int v42; // ecx
  int v43; // ecx
  __int64 *v44; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v45[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = &qword_4FDC280;
  v5 = *(_DWORD *)(a2 + 24);
  v44 = &qword_4FDC280;
  if ( !v5 )
  {
    ++*(_QWORD *)a2;
    v45[0] = 0;
LABEL_75:
    v5 *= 2;
    goto LABEL_76;
  }
  v6 = *(_QWORD *)(a2 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)&qword_4FDC280 >> 9) ^ ((unsigned int)&qword_4FDC280 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = (__int64 *)*v10;
  if ( (__int64 *)*v10 == &qword_4FDC280 )
  {
LABEL_3:
    v12 = v10 + 1;
    goto LABEL_4;
  }
  while ( v11 != (__int64 *)-4096LL )
  {
    if ( !v8 && v11 == (__int64 *)-8192LL )
      v8 = (__int64 **)v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = (__int64 *)*v10;
    if ( (__int64 *)*v10 == &qword_4FDC280 )
      goto LABEL_3;
    ++v7;
  }
  v39 = *(_DWORD *)(a2 + 16);
  if ( !v8 )
    v8 = (__int64 **)v10;
  ++*(_QWORD *)a2;
  v40 = v39 + 1;
  v45[0] = v8;
  if ( 4 * v40 >= 3 * v5 )
    goto LABEL_75;
  if ( v5 - *(_DWORD *)(a2 + 20) - v40 <= v5 >> 3 )
  {
LABEL_76:
    sub_2362DB0(a2, v5);
    sub_2351910(a2, (__int64 *)&v44, v45);
    v4 = v44;
    v8 = (__int64 **)v45[0];
    v40 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v40;
  if ( *v8 != (__int64 *)-4096LL )
    --*(_DWORD *)(a2 + 20);
  *v8 = v4;
  v12 = (__int64 *)(v8 + 1);
  v8[1] = 0;
LABEL_4:
  if ( !*v12 )
  {
    v36 = (_QWORD *)sub_22077B0(0x10u);
    if ( v36 )
      *v36 = &unk_4A0BA48;
    v37 = *v12;
    *v12 = (__int64)v36;
    if ( v37 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v37 + 8LL))(v37);
  }
  v13 = &qword_4FDADA8;
  v14 = *(_DWORD *)(a2 + 24);
  v44 = &qword_4FDADA8;
  if ( !v14 )
  {
    ++*(_QWORD *)a2;
    v45[0] = 0;
LABEL_72:
    v14 *= 2;
    goto LABEL_73;
  }
  v15 = *(_QWORD *)(a2 + 8);
  v16 = 1;
  v17 = 0;
  v18 = (v14 - 1) & (((unsigned int)&qword_4FDADA8 >> 9) ^ ((unsigned int)&qword_4FDADA8 >> 4));
  v19 = (_QWORD *)(v15 + 16LL * v18);
  v20 = (__int64 *)*v19;
  if ( (__int64 *)*v19 == &qword_4FDADA8 )
  {
LABEL_7:
    v21 = v19 + 1;
    goto LABEL_8;
  }
  while ( v20 != (__int64 *)-4096LL )
  {
    if ( v20 == (__int64 *)-8192LL && !v17 )
      v17 = (__int64 **)v19;
    v18 = (v14 - 1) & (v16 + v18);
    v19 = (_QWORD *)(v15 + 16LL * v18);
    v20 = (__int64 *)*v19;
    if ( (__int64 *)*v19 == &qword_4FDADA8 )
      goto LABEL_7;
    ++v16;
  }
  v41 = *(_DWORD *)(a2 + 16);
  if ( !v17 )
    v17 = (__int64 **)v19;
  ++*(_QWORD *)a2;
  v42 = v41 + 1;
  v45[0] = v17;
  if ( 4 * v42 >= 3 * v14 )
    goto LABEL_72;
  if ( v14 - *(_DWORD *)(a2 + 20) - v42 <= v14 >> 3 )
  {
LABEL_73:
    sub_2362DB0(a2, v14);
    sub_2351910(a2, (__int64 *)&v44, v45);
    v13 = v44;
    v17 = (__int64 **)v45[0];
    v42 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v42;
  if ( *v17 != (__int64 *)-4096LL )
    --*(_DWORD *)(a2 + 20);
  *v17 = v13;
  v21 = (__int64 *)(v17 + 1);
  v17[1] = 0;
LABEL_8:
  if ( !*v21 )
  {
    v34 = (_QWORD *)sub_22077B0(0x10u);
    if ( v34 )
      *v34 = &unk_4A0BA78;
    v35 = *v21;
    *v21 = (__int64)v34;
    if ( v35 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
  }
  result = &qword_4F8A320;
  v23 = (__int64 **)*(unsigned int *)(a2 + 24);
  v44 = &qword_4F8A320;
  if ( !(_DWORD)v23 )
  {
    ++*(_QWORD *)a2;
    v45[0] = 0;
LABEL_69:
    LODWORD(v23) = 2 * (_DWORD)v23;
    goto LABEL_70;
  }
  v24 = (unsigned int)((_DWORD)v23 - 1);
  v25 = *(_QWORD *)(a2 + 8);
  v26 = 1;
  v27 = 0;
  v28 = (unsigned int)v24 & (((unsigned int)&qword_4F8A320 >> 9) ^ ((unsigned int)&qword_4F8A320 >> 4));
  v29 = (__int64 **)(v25 + 16 * v28);
  v30 = *v29;
  if ( *v29 == &qword_4F8A320 )
  {
LABEL_11:
    v31 = v29 + 1;
    goto LABEL_12;
  }
  while ( v30 != (__int64 *)-4096LL )
  {
    if ( !v27 && v30 == (__int64 *)-8192LL )
      v27 = v29;
    v28 = (unsigned int)v24 & (v26 + (_DWORD)v28);
    v29 = (__int64 **)(v25 + 16LL * (unsigned int)v28);
    v30 = *v29;
    if ( *v29 == &qword_4F8A320 )
      goto LABEL_11;
    ++v26;
  }
  v43 = *(_DWORD *)(a2 + 16);
  if ( !v27 )
    v27 = v29;
  ++*(_QWORD *)a2;
  v25 = (unsigned int)(v43 + 1);
  v45[0] = v27;
  if ( 4 * (int)v25 >= (unsigned int)(3 * (_DWORD)v23) )
    goto LABEL_69;
  v28 = (unsigned int)((_DWORD)v23 - *(_DWORD *)(a2 + 20) - v25);
  v24 = (unsigned int)v23 >> 3;
  if ( (unsigned int)v28 <= (unsigned int)v24 )
  {
LABEL_70:
    sub_2362DB0(a2, (int)v23);
    v23 = &v44;
    v24 = a2;
    sub_2351910(a2, (__int64 *)&v44, v45);
    result = v44;
    v27 = (__int64 **)v45[0];
    v25 = (unsigned int)(*(_DWORD *)(a2 + 16) + 1);
  }
  *(_DWORD *)(a2 + 16) = v25;
  if ( *v27 != (__int64 *)-4096LL )
    --*(_DWORD *)(a2 + 20);
  *v27 = result;
  v31 = v27 + 1;
  v27[1] = 0;
LABEL_12:
  if ( !*v31 )
  {
    v38 = *(_QWORD *)(a1 + 200);
    result = (__int64 *)sub_22077B0(0x10u);
    if ( result )
    {
      result[1] = v38;
      v28 = (unsigned __int64)&unk_4A0BAA8;
      *result = (__int64)&unk_4A0BAA8;
    }
    v24 = (__int64)*v31;
    *v31 = result;
    if ( v24 )
      result = (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
  }
  v32 = *(_QWORD *)(a1 + 1488);
  for ( i = v32 + 32LL * *(unsigned int *)(a1 + 1496); v32 != i; v32 += 32 )
  {
    if ( !*(_QWORD *)(v32 + 16) )
      sub_4263D6(v24, v23, v28);
    v24 = v32;
    v23 = (__int64 **)a2;
    result = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, __int64, __int64 **))(v32 + 24))(
                          v32,
                          a2,
                          v28,
                          v25,
                          v27);
  }
  return result;
}
