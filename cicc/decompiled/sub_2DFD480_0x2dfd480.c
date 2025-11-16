// Function: sub_2DFD480
// Address: 0x2dfd480
//
void __fastcall sub_2DFD480(__int64 a1, __int64 a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r14
  const __m128i *v9; // r14
  __int32 v10; // edi
  int v11; // ebx
  unsigned int v12; // r15d
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rdx
  int *v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 *v24; // r10
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __m128i *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned int v34; // ebx
  _BYTE *v35; // rdi
  char v36; // al
  __int64 v37; // r12
  __int64 v38; // r9
  __int8 *v39; // r12
  __int64 v40; // [rsp+8h] [rbp-148h]
  char v41; // [rsp+10h] [rbp-140h]
  char v42; // [rsp+14h] [rbp-13Ch]
  const __m128i *v44; // [rsp+28h] [rbp-128h]
  const __m128i *v45; // [rsp+30h] [rbp-120h]
  unsigned __int64 v46[4]; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v47[4]; // [rsp+60h] [rbp-F0h] BYREF
  int *v48; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v49; // [rsp+88h] [rbp-C8h]
  _BYTE v50[48]; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+C0h] [rbp-90h] BYREF
  _QWORD *v52; // [rsp+C8h] [rbp-88h]
  __int64 v53; // [rsp+D0h] [rbp-80h]
  _QWORD v54[15]; // [rsp+D8h] [rbp-78h] BYREF

  v7 = a1;
  v48 = (int *)v50;
  v49 = 0xC00000000LL;
  v42 = a5;
  v41 = a6;
  v44 = (const __m128i *)((char *)a3 + 40 * a4);
  if ( a3 != v44 )
  {
    v9 = a3;
    v40 = a1 + 56;
    while ( 1 )
    {
      v45 = v9;
      if ( v9->m128i_i8[0] )
        break;
      v10 = v9->m128i_i32[2];
      v11 = -1;
      if ( v10 )
      {
        v12 = *(_DWORD *)(a1 + 64);
        v13 = *(_QWORD *)(a1 + 56);
        if ( v12 )
        {
          v14 = *(_QWORD *)(a1 + 56);
          v11 = 0;
          while ( *(_BYTE *)v14
               || v10 != *(_DWORD *)(v14 + 8)
               || ((*(_DWORD *)v14 >> 8) & 0xFFF) != (((unsigned __int32)v9->m128i_i32[0] >> 8) & 0xFFF) )
          {
            ++v11;
            v14 += 40;
            if ( v12 == v11 )
              goto LABEL_35;
          }
          goto LABEL_11;
        }
        goto LABEL_35;
      }
LABEL_11:
      v15 = (unsigned int)v49;
      v16 = (unsigned int)v49 + 1LL;
      if ( v16 > HIDWORD(v49) )
      {
        sub_C8D5F0((__int64)&v48, v50, v16, 4u, a5, a6);
        v15 = (unsigned int)v49;
      }
      v9 = (const __m128i *)((char *)v9 + 40);
      v48[v15] = v11;
      v17 = (unsigned int)(v49 + 1);
      LODWORD(v49) = v49 + 1;
      if ( v44 == v9 )
      {
        v18 = v48;
        v7 = a1;
        goto LABEL_15;
      }
    }
    v12 = *(_DWORD *)(a1 + 64);
    v37 = 0;
    v11 = 0;
    if ( v12 )
    {
      while ( !(unsigned __int8)sub_2EAB6C0(v9, v37 + *(_QWORD *)(a1 + 56)) )
      {
        ++v11;
        v37 += 40;
        if ( v12 == v11 )
        {
          v12 = *(_DWORD *)(a1 + 64);
          goto LABEL_34;
        }
      }
      goto LABEL_11;
    }
LABEL_34:
    v13 = *(_QWORD *)(a1 + 56);
LABEL_35:
    v29 = v12;
    v30 = v12 + 1LL;
    if ( v30 > *(unsigned int *)(a1 + 68) )
    {
      v38 = a1 + 72;
      if ( v13 > (unsigned __int64)v9 || v13 + 40LL * v12 <= (unsigned __int64)v9 )
      {
        sub_C8D5F0(v40, (const void *)(a1 + 72), v30, 0x28u, a5, v38);
        v13 = *(_QWORD *)(a1 + 56);
        v29 = *(unsigned int *)(a1 + 64);
      }
      else
      {
        v39 = &v9->m128i_i8[-v13];
        sub_C8D5F0(v40, (const void *)(a1 + 72), v30, 0x28u, a5, v38);
        v13 = *(_QWORD *)(a1 + 56);
        v45 = (const __m128i *)&v39[v13];
        v29 = *(unsigned int *)(a1 + 64);
      }
    }
    v31 = (__m128i *)(v13 + 40 * v29);
    *v31 = _mm_loadu_si128(v45);
    v31[1] = _mm_loadu_si128(v45 + 1);
    v31[2].m128i_i64[0] = v45[2].m128i_i64[0];
    v32 = *(_QWORD *)(a1 + 56);
    v33 = (unsigned int)(*(_DWORD *)(a1 + 64) + 1);
    *(_DWORD *)(a1 + 64) = v33;
    *(_QWORD *)(v32 + 40 * v33 - 24) = 0;
    v34 = *(_DWORD *)(a1 + 64);
    v35 = (_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL * v34 - 40);
    if ( !*v35 )
    {
      v36 = v35[3];
      if ( (v36 & 0x10) != 0 )
      {
        v35[3] = v36 & 0xBF;
        v35 = (_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL * *(unsigned int *)(a1 + 64) - 40);
      }
      sub_2EAB250(v35, 0);
      v34 = *(_DWORD *)(a1 + 64);
    }
    v11 = v34 - 1;
    goto LABEL_11;
  }
  v18 = (int *)v50;
  v17 = 0;
LABEL_15:
  sub_2DF5BF0((__int64)v46, v18, v17, v42, v41, a7);
  v22 = *(unsigned int *)(v7 + 392);
  v51 = v7 + 232;
  v52 = v54;
  v53 = 0x400000000LL;
  if ( (_DWORD)v22 )
  {
    sub_2DF6390((__int64)&v51, a2, v22, v19, v20, v21);
    if ( !(_DWORD)v53 )
      goto LABEL_22;
  }
  else
  {
    v23 = *(unsigned int *)(v7 + 396);
    if ( (_DWORD)v23 )
    {
      v24 = (__int64 *)(v7 + 240);
      do
      {
        if ( (*(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v24 >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                | (unsigned int)(a2 >> 1)
                                                                                                & 3) )
          break;
        v22 = (unsigned int)(v22 + 1);
        v24 += 2;
      }
      while ( (_DWORD)v23 != (_DWORD)v22 );
    }
    v54[0] = v7 + 232;
    LODWORD(v53) = 1;
    v54[1] = v23 | (v22 << 32);
  }
  if ( *((_DWORD *)v52 + 3) < *((_DWORD *)v52 + 2) && *(_QWORD *)sub_2DF4990((__int64)&v51) == a2 )
  {
    sub_2DF52D0((__int64)v47, (__int64)v46);
    sub_2DF7680((__int64)&v51, (__int64)v47);
    v28 = v47[0];
    if ( !v47[0] )
      goto LABEL_26;
LABEL_25:
    j_j___libc_free_0_0(v28);
    goto LABEL_26;
  }
LABEL_22:
  sub_2DF52D0((__int64)v47, (__int64)v46);
  v25 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v26 = (a2 >> 1) & 3;
  if ( v26 == 3 )
    v27 = *(_QWORD *)(v25 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  else
    v27 = v25 | (2 * v26 + 2);
  sub_2DFCEE0((__int64)&v51, a2, v27, (__int64)v47);
  v28 = v47[0];
  if ( v47[0] )
    goto LABEL_25;
LABEL_26:
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  if ( v46[0] )
    j_j___libc_free_0_0(v46[0]);
  if ( v48 != (int *)v50 )
    _libc_free((unsigned __int64)v48);
}
