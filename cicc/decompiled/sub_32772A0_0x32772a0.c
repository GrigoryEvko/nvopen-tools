// Function: sub_32772A0
// Address: 0x32772a0
//
__int64 __fastcall sub_32772A0(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  int v8; // edi
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rax
  int v13; // r12d
  __int64 result; // rax
  int v15; // r13d
  __int64 v16; // rax
  __int128 v17; // xmm1
  __int64 v18; // rdi
  int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r10
  __int64 v24; // r11
  __int64 v25; // rsi
  int v26; // r15d
  __int64 v27; // r8
  __int128 v28; // rax
  __int128 v29; // rax
  __int64 v30; // rsi
  __int128 v31; // [rsp-C8h] [rbp-C8h]
  __int128 v32; // [rsp-B8h] [rbp-B8h]
  int v33; // [rsp-A0h] [rbp-A0h]
  __int64 v34; // [rsp-98h] [rbp-98h]
  __int64 v35; // [rsp-90h] [rbp-90h]
  int v36; // [rsp-88h] [rbp-88h]
  __int128 v37; // [rsp-88h] [rbp-88h]
  __m128i v38; // [rsp-68h] [rbp-68h]
  int v39; // [rsp-68h] [rbp-68h]
  __int128 v40; // [rsp-58h] [rbp-58h]
  int v41; // [rsp-58h] [rbp-58h]
  __int64 v42; // [rsp-58h] [rbp-58h]
  __int64 v43; // [rsp-48h] [rbp-48h] BYREF
  int v44; // [rsp-40h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 56);
  if ( !v6 )
    return 0;
  v8 = 1;
  do
  {
    while ( a3 != *(_DWORD *)(v6 + 8) )
    {
      v6 = *(_QWORD *)(v6 + 32);
      if ( !v6 )
        goto LABEL_9;
    }
    if ( !v8 )
      return 0;
    v9 = *(_QWORD *)(v6 + 32);
    if ( !v9 )
      goto LABEL_10;
    if ( a3 == *(_DWORD *)(v9 + 8) )
      return 0;
    v6 = *(_QWORD *)(v9 + 32);
    v8 = 0;
  }
  while ( v6 );
LABEL_9:
  if ( v8 == 1 )
    return 0;
LABEL_10:
  v10 = *(_QWORD *)(a4 + 56);
  if ( !v10 )
    return 0;
  v11 = 1;
  do
  {
    while ( a5 != *(_DWORD *)(v10 + 8) )
    {
      v10 = *(_QWORD *)(v10 + 32);
      if ( !v10 )
        goto LABEL_18;
    }
    if ( !v11 )
      return 0;
    v12 = *(_QWORD *)(v10 + 32);
    if ( !v12 )
      goto LABEL_19;
    if ( a5 == *(_DWORD *)(v12 + 8) )
      return 0;
    v10 = *(_QWORD *)(v12 + 32);
    v11 = 0;
  }
  while ( v10 );
LABEL_18:
  if ( v11 == 1 )
    return 0;
LABEL_19:
  v13 = *(_DWORD *)(a1 + 24);
  if ( *(_DWORD *)(a2 + 24) != v13 )
    return 0;
  v15 = *(_DWORD *)(a4 + 24);
  if ( (unsigned int)(v15 - 190) > 2 )
    return 0;
  v16 = *(_QWORD *)(a4 + 40);
  v17 = (__int128)_mm_loadu_si128((const __m128i *)(v16 + 40));
  v18 = *(_QWORD *)(v16 + 40);
  v19 = *(_DWORD *)(v16 + 48);
  v38 = _mm_loadu_si128((const __m128i *)v16);
  v20 = *(__int64 **)(a2 + 40);
  if ( v15 == *(_DWORD *)(*v20 + 24) )
  {
    v30 = *(_QWORD *)(*v20 + 40);
    if ( *(_QWORD *)(v30 + 40) == v18 && *(_DWORD *)(v30 + 48) == v19 )
    {
      v23 = v20[5];
      *(_QWORD *)&v40 = *(_QWORD *)v30;
      *((_QWORD *)&v40 + 1) = *(unsigned int *)(v30 + 8);
      v24 = *((unsigned int *)v20 + 12);
      goto LABEL_28;
    }
  }
  v21 = v20[5];
  if ( v15 != *(_DWORD *)(v21 + 24) )
    return 0;
  v22 = *(_QWORD *)(v21 + 40);
  if ( *(_QWORD *)(v22 + 40) != v18 || *(_DWORD *)(v22 + 48) != v19 )
    return 0;
  v23 = *v20;
  *(_QWORD *)&v40 = *(_QWORD *)v22;
  *((_QWORD *)&v40 + 1) = *(unsigned int *)(v22 + 8);
  v24 = *((unsigned int *)v20 + 2);
LABEL_28:
  v25 = *(_QWORD *)(a1 + 80);
  v26 = **(unsigned __int16 **)(a1 + 48);
  v27 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
  v43 = v25;
  if ( v25 )
  {
    v33 = a6;
    v34 = v23;
    v35 = v24;
    v36 = v27;
    sub_B96E90((__int64)&v43, v25, 1);
    a6 = v33;
    v23 = v34;
    v24 = v35;
    LODWORD(v27) = v36;
  }
  v32 = (__int128)v38;
  v31 = v40;
  *(_QWORD *)&v37 = v23;
  *((_QWORD *)&v37 + 1) = v24;
  v39 = v27;
  v41 = a6;
  v44 = *(_DWORD *)(a1 + 72);
  *(_QWORD *)&v28 = sub_3406EB0(a6, v13, (unsigned int)&v43, v26, v27, a6, v31, v32);
  *(_QWORD *)&v29 = sub_3406EB0(v41, v15, (unsigned int)&v43, v26, v39, v41, v28, v17);
  result = sub_3406EB0(v41, v13, (unsigned int)&v43, v26, v39, v41, v29, v37);
  if ( v43 )
  {
    v42 = result;
    sub_B91220((__int64)&v43, v43);
    return v42;
  }
  return result;
}
