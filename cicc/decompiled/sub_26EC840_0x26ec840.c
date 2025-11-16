// Function: sub_26EC840
// Address: 0x26ec840
//
void __fastcall sub_26EC840(__int64 a1, __int64 a2, __int64 a3)
{
  const char *v4; // r13
  size_t v5; // rdx
  size_t v6; // r15
  int v7; // eax
  __int64 v8; // rdx
  __int64 *v9; // rcx
  __int64 v10; // rbx
  __int64 *v11; // r15
  char v12; // r13
  _QWORD *v13; // r12
  _WORD *v14; // rdi
  float v15; // r14d
  unsigned __int64 v16; // rsi
  float **v17; // rax
  float *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rdi
  __m128i si128; // xmm0
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __m128i *v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rdi
  _BYTE *v36; // rax
  _QWORD *v37; // rax
  unsigned __int64 v38; // r14
  __int64 v39; // rax
  __m128i v40; // xmm3
  unsigned __int64 v41; // rcx
  float **v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r13
  __int64 v46; // rax
  unsigned int v47; // r8d
  __int64 *v48; // rcx
  __int64 v49; // rbx
  __int64 *v50; // rax
  __int64 *v51; // rax
  const char *v52; // rax
  size_t v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  _WORD *v57; // rdx
  unsigned __int64 v59; // [rsp+10h] [rbp-90h]
  size_t v60; // [rsp+10h] [rbp-90h]
  float v61; // [rsp+20h] [rbp-80h]
  unsigned __int64 v62; // [rsp+20h] [rbp-80h]
  __int64 *v63; // [rsp+20h] [rbp-80h]
  __int64 v64; // [rsp+28h] [rbp-78h]
  unsigned int v65; // [rsp+28h] [rbp-78h]
  _QWORD v66[2]; // [rsp+30h] [rbp-70h] BYREF
  float v67; // [rsp+40h] [rbp-60h]
  _QWORD v68[2]; // [rsp+50h] [rbp-50h] BYREF
  float v69; // [rsp+60h] [rbp-40h]

  v4 = sub_BD5D20(a2);
  v6 = v5;
  v7 = sub_C92610();
  v8 = (unsigned int)sub_C92740(a1, v4, v6, v7);
  v9 = (__int64 *)(*(_QWORD *)a1 + 8 * v8);
  v10 = *v9;
  if ( *v9 )
  {
    if ( v10 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 16);
  }
  v63 = v9;
  v65 = v8;
  v46 = sub_C7D670(v6 + 65, 8);
  v47 = v65;
  v48 = v63;
  v49 = v46;
  if ( v6 )
  {
    memcpy((void *)(v46 + 64), v4, v6);
    v47 = v65;
    v48 = v63;
  }
  *(_BYTE *)(v49 + v6 + 64) = 0;
  *(_OWORD *)(v49 + 40) = 0;
  *(_QWORD *)v49 = v6;
  *(_QWORD *)(v49 + 56) = 0;
  *(_QWORD *)(v49 + 8) = v49 + 56;
  *(_QWORD *)(v49 + 16) = 1;
  *(_DWORD *)(v49 + 40) = 1065353216;
  *(_QWORD *)(v49 + 48) = 0;
  *(_OWORD *)(v49 + 24) = 0;
  *v48 = v49;
  ++*(_DWORD *)(a1 + 12);
  v50 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v47));
  v10 = *v50;
  if ( *v50 == -8 || !v10 )
  {
    v51 = v50 + 1;
    do
    {
      do
        v10 = *v51++;
      while ( !v10 );
    }
    while ( v10 == -8 );
  }
LABEL_3:
  v11 = *(__int64 **)(a3 + 16);
  v12 = 0;
  v13 = (_QWORD *)(v10 + 8);
  if ( v11 )
  {
    v64 = v10;
    do
    {
      v14 = v13;
      v15 = *((float *)v11 + 6);
      v16 = (unsigned __int64)(v11[2] + 31 * v11[1]) % *(_QWORD *)(v64 + 16);
      v17 = (float **)sub_26EC1A0(v13, v16, v11 + 1, v11[2] + 31 * v11[1]);
      if ( v17 && (v18 = *v17) != 0 )
      {
        v61 = v18[6];
        if ( fabs(v15 - v61) > 0.02 )
        {
          if ( !v12 )
          {
            v43 = sub_C5F790((__int64)v13, v16);
            v44 = *(_QWORD *)(v43 + 32);
            v45 = v43;
            if ( (unsigned __int64)(*(_QWORD *)(v43 + 24) - v44) <= 8 )
            {
              v45 = sub_CB6200(v43, (unsigned __int8 *)"Function ", 9u);
            }
            else
            {
              *(_BYTE *)(v44 + 8) = 32;
              *(_QWORD *)v44 = 0x6E6F6974636E7546LL;
              *(_QWORD *)(v43 + 32) += 9LL;
            }
            v52 = sub_BD5D20(a2);
            v14 = *(_WORD **)(v45 + 32);
            v16 = (unsigned __int64)v52;
            v54 = *(_QWORD *)(v45 + 24) - (_QWORD)v14;
            if ( v54 < v53 )
            {
              v55 = sub_CB6200(v45, (unsigned __int8 *)v16, v53);
              v14 = *(_WORD **)(v55 + 32);
              v45 = v55;
              v54 = *(_QWORD *)(v55 + 24) - (_QWORD)v14;
            }
            else if ( v53 )
            {
              v60 = v53;
              memcpy(v14, (const void *)v16, v53);
              v56 = *(_QWORD *)(v45 + 24);
              v57 = (_WORD *)(*(_QWORD *)(v45 + 32) + v60);
              *(_QWORD *)(v45 + 32) = v57;
              v14 = v57;
              v54 = v56 - (_QWORD)v57;
            }
            if ( v54 <= 1 )
            {
              v16 = (unsigned __int64)":\n";
              v14 = (_WORD *)v45;
              sub_CB6200(v45, (unsigned __int8 *)":\n", 2u);
            }
            else
            {
              *v14 = 2618;
              *(_QWORD *)(v45 + 32) += 2LL;
            }
          }
          v19 = sub_C5F790((__int64)v14, v16);
          v20 = *(_QWORD *)(v19 + 32);
          v21 = v19;
          if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v20) <= 5 )
          {
            v21 = sub_CB6200(v19, "Probe ", 6u);
          }
          else
          {
            *(_DWORD *)v20 = 1651470928;
            *(_WORD *)(v20 + 4) = 8293;
            *(_QWORD *)(v19 + 32) += 6LL;
          }
          v22 = sub_CB59D0(v21, v11[1]);
          v26 = *(_QWORD *)(v22 + 32);
          v27 = v22;
          if ( (unsigned __int64)(*(_QWORD *)(v22 + 24) - v26) <= 0x10 )
          {
            v27 = sub_CB6200(v22, "\tprevious factor ", 0x11u);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_4391BE0);
            *(_BYTE *)(v26 + 16) = 32;
            *(__m128i *)v26 = si128;
            *(_QWORD *)(v22 + 32) += 17LL;
          }
          v66[1] = "%0.2f";
          v67 = v61;
          v66[0] = &unk_49DB348;
          v29 = sub_CB6620(v27, (__int64)v66, v26, v23, v24, v25);
          v33 = *(__m128i **)(v29 + 32);
          v34 = v29;
          if ( *(_QWORD *)(v29 + 24) - (_QWORD)v33 <= 0xFu )
          {
            v34 = sub_CB6200(v29, "\tcurrent factor ", 0x10u);
          }
          else
          {
            *v33 = _mm_load_si128((const __m128i *)&xmmword_4391BF0);
            *(_QWORD *)(v29 + 32) += 16LL;
          }
          v69 = v15;
          v68[1] = "%0.2f";
          v68[0] = &unk_49DB348;
          v35 = sub_CB6620(v34, (__int64)v68, (__int64)v33, v30, v31, v32);
          v36 = *(_BYTE **)(v35 + 32);
          if ( *(_BYTE **)(v35 + 24) == v36 )
          {
            sub_CB6200(v35, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v36 = 10;
            ++*(_QWORD *)(v35 + 32);
          }
          v15 = *((float *)v11 + 6);
          v12 = 1;
        }
      }
      else
      {
        v37 = (_QWORD *)sub_22077B0(0x28u);
        v38 = (unsigned __int64)v37;
        if ( v37 )
          *v37 = 0;
        v39 = v11[1];
        v40 = _mm_loadu_si128((const __m128i *)(v11 + 1));
        *(_DWORD *)(v38 + 24) = 0;
        *(__m128i *)(v38 + 8) = v40;
        v41 = *(_QWORD *)(v38 + 16) + 31 * v39;
        v59 = v41;
        v62 = v41 % *(_QWORD *)(v64 + 16);
        v42 = (float **)sub_26EC1A0(v13, v62, (_QWORD *)(v38 + 8), v41);
        if ( v42 && (v18 = *v42) != 0 )
          j_j___libc_free_0(v38);
        else
          v18 = (float *)sub_26E92A0(v13, v62, v59, (_QWORD *)v38, 1);
        v15 = *((float *)v11 + 6);
      }
      v18[6] = v15;
      v11 = (__int64 *)*v11;
    }
    while ( v11 );
  }
}
