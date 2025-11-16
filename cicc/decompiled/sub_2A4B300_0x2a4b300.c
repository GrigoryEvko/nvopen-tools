// Function: sub_2A4B300
// Address: 0x2a4b300
//
__int64 __fastcall sub_2A4B300(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r15
  __m128i *v12; // rdx
  __m128i si128; // xmm0
  const char *v14; // rax
  size_t v15; // rdx
  __int64 v16; // r8
  unsigned __int8 *v17; // rsi
  _BYTE *v18; // rax
  _BYTE *v19; // rdi
  __int64 v20; // rax
  __int64 *v21; // r8
  __int64 v22; // rsi
  __int64 v23; // r14
  __int64 *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // rbx
  __int64 v28; // r8
  __int64 *i; // r13
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // rax
  __int64 v34; // r11
  __int64 v35; // rax
  int v36; // eax
  _BYTE *v37; // rax
  int v38; // r10d
  __int64 v39; // [rsp+0h] [rbp-40h]
  __int64 v40; // [rsp+0h] [rbp-40h]
  __int64 v41; // [rsp+8h] [rbp-38h]
  __int64 *v42; // [rsp+8h] [rbp-38h]
  __int64 *v43; // [rsp+8h] [rbp-38h]
  unsigned __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 *v45; // [rsp+8h] [rbp-38h]
  size_t v46; // [rsp+8h] [rbp-38h]

  v8 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v9 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v10 = *a2;
  v11 = v9 + 8;
  v12 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v12 <= 0x1Bu )
  {
    v10 = sub_CB6200(*a2, "PredicateInfo for function: ", 0x1Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42C77B0);
    qmemcpy(&v12[1], "r function: ", 12);
    *v12 = si128;
    *(_QWORD *)(v10 + 32) += 28LL;
  }
  v41 = v10;
  v14 = sub_BD5D20(a3);
  v16 = v41;
  v17 = (unsigned __int8 *)v14;
  v18 = *(_BYTE **)(v41 + 24);
  v19 = *(_BYTE **)(v41 + 32);
  if ( v18 - v19 < v15 )
  {
    v16 = sub_CB6200(v41, v17, v15);
    v18 = *(_BYTE **)(v16 + 24);
    v19 = *(_BYTE **)(v16 + 32);
  }
  else if ( v15 )
  {
    v40 = v41;
    v46 = v15;
    memcpy(v19, v17, v15);
    v16 = v40;
    v37 = *(_BYTE **)(v40 + 24);
    v19 = (_BYTE *)(*(_QWORD *)(v40 + 32) + v46);
    *(_QWORD *)(v40 + 32) = v19;
    if ( v37 != v19 )
      goto LABEL_6;
LABEL_42:
    sub_CB6200(v16, (unsigned __int8 *)"\n", 1u);
    goto LABEL_7;
  }
  if ( v18 == v19 )
    goto LABEL_42;
LABEL_6:
  *v19 = 10;
  ++*(_QWORD *)(v16 + 32);
LABEL_7:
  v20 = sub_22077B0(0x258u);
  v21 = (__int64 *)v20;
  if ( v20 )
  {
    v42 = (__int64 *)v20;
    sub_2A4B140(v20, a3, v8, v11);
    v21 = v42;
  }
  v22 = *a2;
  v43 = v21;
  v23 = a3 + 72;
  sub_2A45920(v21, *a2);
  v27 = *(_QWORD *)(a3 + 80);
  v28 = (__int64)v43;
  if ( v23 != v27 )
  {
    if ( !v27 )
      BUG();
    while ( 1 )
    {
      v24 = *(__int64 **)(v27 + 32);
      if ( v24 != (__int64 *)(v27 + 24) )
        break;
      v27 = *(_QWORD *)(v27 + 8);
      if ( v23 == v27 )
        goto LABEL_15;
      if ( !v27 )
        BUG();
    }
    if ( v23 != v27 )
    {
      while ( 1 )
      {
        for ( i = (__int64 *)v24[1]; ; i = *(__int64 **)(v27 + 32) )
        {
          v31 = v27 - 24;
          if ( !v27 )
            v31 = 0;
          if ( i != (__int64 *)(v31 + 48) )
            break;
          v27 = *(_QWORD *)(v27 + 8);
          if ( v23 == v27 )
            break;
          if ( !v27 )
            BUG();
        }
        v22 = *(unsigned int *)(v28 + 48);
        v26 = *(_QWORD *)(v28 + 32);
        v32 = (__int64)(v24 - 3);
        if ( !(_DWORD)v22 )
          goto LABEL_33;
        v22 = (unsigned int)(v22 - 1);
        v25 = (unsigned int)v22 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
        v33 = (__int64 *)(v26 + 16 * v25);
        v34 = *v33;
        if ( v32 != *v33 )
          break;
LABEL_28:
        v35 = v33[1];
        if ( *((_BYTE *)v24 - 24) != 85 )
          goto LABEL_29;
LABEL_34:
        v25 = *(v24 - 7);
        if ( v25 )
        {
          if ( !*(_BYTE *)v25 )
          {
            v22 = v24[7];
            if ( *(_QWORD *)(v25 + 24) == v22
              && (*(_BYTE *)(v25 + 33) & 0x20) != 0
              && v35
              && *(_DWORD *)(v25 + 36) == 336 )
            {
              v39 = v28;
              v45 = v24 - 3;
              v22 = *(_QWORD *)(v32 - 32LL * (*((_DWORD *)v24 - 5) & 0x7FFFFFF));
              sub_BD84D0(v32, v22);
              sub_B43D60(v45);
              v28 = v39;
            }
          }
        }
LABEL_29:
        if ( v23 == v27 )
          goto LABEL_15;
        v24 = i;
      }
      v36 = 1;
      while ( v34 != -4096 )
      {
        v38 = v36 + 1;
        v25 = (unsigned int)v22 & (v36 + (_DWORD)v25);
        v33 = (__int64 *)(v26 + 16LL * (unsigned int)v25);
        v34 = *v33;
        if ( v32 == *v33 )
          goto LABEL_28;
        v36 = v38;
      }
LABEL_33:
      v35 = 0;
      if ( *((_BYTE *)v24 - 24) != 85 )
        goto LABEL_29;
      goto LABEL_34;
    }
  }
LABEL_15:
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  if ( v28 )
  {
    v44 = v28;
    sub_2A45460(v28, v22, v24, v25, v28, v26);
    j_j___libc_free_0(v44);
  }
  return a1;
}
