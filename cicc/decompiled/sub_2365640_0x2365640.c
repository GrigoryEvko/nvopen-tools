// Function: sub_2365640
// Address: 0x2365640
//
__int64 *__fastcall sub_2365640(__int64 a1, __int64 a2)
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
  __int64 *v13; // rbx
  __int64 *v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rcx
  int v17; // r11d
  __int64 **v18; // r8
  unsigned int v19; // edx
  _QWORD *v20; // rbx
  __int64 *v21; // r9
  __int64 *v22; // rbx
  __int64 *v23; // rbx
  __int64 *v24; // rsi
  __int64 v25; // rdi
  __int64 *result; // rax
  void *v27; // rdx
  __int64 *v28; // rbx
  __int64 v29; // rbx
  __int64 i; // r13
  _QWORD *v31; // rax
  __int64 v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r15
  _QWORD *v36; // rax
  __int64 v37; // rdi
  _QWORD *v38; // rax
  __int64 v39; // rdi
  int v40; // ecx
  int v41; // ecx
  int v42; // ecx
  int v43; // ecx
  __int64 *v44; // [rsp+0h] [rbp-40h] BYREF
  __int64 v45[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = qword_502E9E0;
  v5 = *(_DWORD *)(a2 + 24);
  v44 = qword_502E9E0;
  if ( !v5 )
  {
    ++*(_QWORD *)a2;
    v45[0] = 0;
LABEL_66:
    sub_2364D40(a2, 2 * v5);
LABEL_67:
    sub_2351B50(a2, (__int64 *)&v44, v45);
    v4 = v44;
    v8 = (__int64 **)v45[0];
    v41 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_46;
  }
  v6 = *(_QWORD *)(a2 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)qword_502E9E0 >> 9) ^ ((unsigned int)qword_502E9E0 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = (__int64 *)*v10;
  if ( (__int64 *)*v10 == qword_502E9E0 )
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
    if ( (__int64 *)*v10 == qword_502E9E0 )
      goto LABEL_3;
    ++v7;
  }
  v40 = *(_DWORD *)(a2 + 16);
  if ( !v8 )
    v8 = (__int64 **)v10;
  ++*(_QWORD *)a2;
  v41 = v40 + 1;
  v45[0] = (__int64)v8;
  if ( 4 * v41 >= 3 * v5 )
    goto LABEL_66;
  if ( v5 - *(_DWORD *)(a2 + 20) - v41 <= v5 >> 3 )
  {
    sub_2364D40(a2, v5);
    goto LABEL_67;
  }
LABEL_46:
  *(_DWORD *)(a2 + 16) = v41;
  if ( *v8 != (__int64 *)-4096LL )
    --*(_DWORD *)(a2 + 20);
  *v8 = v4;
  v12 = (__int64 *)(v8 + 1);
  v8[1] = 0;
LABEL_4:
  if ( !*v12 )
  {
    v33 = (_QWORD *)sub_22077B0(0x10u);
    if ( v33 )
      *v33 = &unk_4A0C288;
    v34 = *v12;
    *v12 = (__int64)v33;
    if ( v34 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
  }
  v45[0] = (__int64)&unk_4FDB6B0;
  v13 = sub_2364F40(a2, v45);
  if ( !*v13 )
  {
    v31 = (_QWORD *)sub_22077B0(0x10u);
    if ( v31 )
      *v31 = &unk_4A0C2B8;
    v32 = *v13;
    *v13 = (__int64)v31;
    if ( v32 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
  }
  v14 = &qword_4FDC270;
  v15 = *(_DWORD *)(a2 + 24);
  v44 = &qword_4FDC270;
  if ( !v15 )
  {
    ++*(_QWORD *)a2;
    v45[0] = 0;
LABEL_63:
    v15 *= 2;
    goto LABEL_64;
  }
  v16 = *(_QWORD *)(a2 + 8);
  v17 = 1;
  v18 = 0;
  v19 = (v15 - 1) & (((unsigned int)&qword_4FDC270 >> 9) ^ ((unsigned int)&qword_4FDC270 >> 4));
  v20 = (_QWORD *)(v16 + 16LL * v19);
  v21 = (__int64 *)*v20;
  if ( (__int64 *)*v20 == &qword_4FDC270 )
  {
LABEL_8:
    v22 = v20 + 1;
    goto LABEL_9;
  }
  while ( v21 != (__int64 *)-4096LL )
  {
    if ( !v18 && v21 == (__int64 *)-8192LL )
      v18 = (__int64 **)v20;
    v19 = (v15 - 1) & (v17 + v19);
    v20 = (_QWORD *)(v16 + 16LL * v19);
    v21 = (__int64 *)*v20;
    if ( (__int64 *)*v20 == &qword_4FDC270 )
      goto LABEL_8;
    ++v17;
  }
  v42 = *(_DWORD *)(a2 + 16);
  if ( !v18 )
    v18 = (__int64 **)v20;
  ++*(_QWORD *)a2;
  v43 = v42 + 1;
  v45[0] = (__int64)v18;
  if ( 4 * v43 >= 3 * v15 )
    goto LABEL_63;
  if ( v15 - *(_DWORD *)(a2 + 20) - v43 <= v15 >> 3 )
  {
LABEL_64:
    sub_2364D40(a2, v15);
    sub_2351B50(a2, (__int64 *)&v44, v45);
    v14 = v44;
    v18 = (__int64 **)v45[0];
    v43 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v43;
  if ( *v18 != (__int64 *)-4096LL )
    --*(_DWORD *)(a2 + 20);
  *v18 = v14;
  v22 = (__int64 *)(v18 + 1);
  v18[1] = 0;
LABEL_9:
  if ( !*v22 )
  {
    v38 = (_QWORD *)sub_22077B0(0x10u);
    if ( v38 )
      *v38 = &unk_4A0C2E8;
    v39 = *v22;
    *v22 = (__int64)v38;
    if ( v39 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 8LL))(v39);
  }
  v45[0] = (__int64)&qword_4F8A320;
  v23 = sub_2364F40(a2, v45);
  if ( !*v23 )
  {
    v35 = *(_QWORD *)(a1 + 200);
    v36 = (_QWORD *)sub_22077B0(0x10u);
    if ( v36 )
    {
      v36[1] = v35;
      *v36 = &unk_4A0C318;
    }
    v37 = *v23;
    *v23 = (__int64)v36;
    if ( v37 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v37 + 8LL))(v37);
  }
  v24 = v45;
  v25 = a2;
  v45[0] = (__int64)qword_50059C8;
  result = sub_2364F40(a2, v45);
  v28 = result;
  if ( !*result )
  {
    result = (__int64 *)sub_22077B0(0x10u);
    if ( result )
    {
      v27 = &unk_4A0C348;
      *result = (__int64)&unk_4A0C348;
    }
    v25 = *v28;
    *v28 = (__int64)result;
    if ( v25 )
      result = (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
  }
  v29 = *(_QWORD *)(a1 + 1808);
  for ( i = v29 + 32LL * *(unsigned int *)(a1 + 1816); v29 != i; v29 += 32 )
  {
    if ( !*(_QWORD *)(v29 + 16) )
      sub_4263D6(v25, v24, v27);
    v25 = v29;
    v24 = (__int64 *)a2;
    result = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(v29 + 24))(v29, a2);
  }
  return result;
}
