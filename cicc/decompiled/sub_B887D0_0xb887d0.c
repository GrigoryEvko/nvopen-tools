// Function: sub_B887D0
// Address: 0xb887d0
//
__int64 __fastcall sub_B887D0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 *v4; // r13
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 *v7; // r15
  __int64 *i; // r14
  __int64 v9; // rbx
  __int64 *v10; // r15
  __int64 v11; // r8
  __int64 *v12; // r14
  __int64 *j; // r13
  _QWORD *v14; // rdi
  _QWORD *v15; // rsi
  _QWORD *v16; // rdi
  _QWORD *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rax
  size_t v22; // rdx
  __int64 v23; // r8
  const char *v24; // rsi
  __m128i *v25; // rdi
  unsigned __int64 v26; // rax
  __m128i v27; // xmm0
  __int64 v28; // rax
  size_t v29; // rdx
  __int64 v30; // r8
  const void *v31; // rsi
  _WORD *v32; // rdi
  unsigned __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // rax
  __int64 v36; // rax
  size_t v37; // rdx
  __int64 v38; // r8
  const char *v39; // rsi
  __m128i *v40; // rdi
  unsigned __int64 v41; // rax
  __m128i si128; // xmm0
  __int64 v43; // rax
  size_t v44; // rdx
  __int64 v45; // r8
  const void *v46; // rsi
  _WORD *v47; // rdi
  unsigned __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __m128i *v52; // rdx
  __int64 v53; // rax
  _WORD *v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __m128i *v58; // rdx
  __int64 v59; // rax
  _WORD *v60; // rdx
  __int64 v61; // [rsp+0h] [rbp-60h]
  __int64 v62; // [rsp+8h] [rbp-58h]
  __int64 v63; // [rsp+8h] [rbp-58h]
  size_t v64; // [rsp+8h] [rbp-58h]
  __int64 v65; // [rsp+10h] [rbp-50h]
  size_t v66; // [rsp+10h] [rbp-50h]
  __int64 v67; // [rsp+10h] [rbp-50h]
  __int64 *v69; // [rsp+20h] [rbp-40h]
  __int64 v70; // [rsp+20h] [rbp-40h]
  __int64 v71; // [rsp+20h] [rbp-40h]
  size_t v72; // [rsp+20h] [rbp-40h]
  __int64 *v73; // [rsp+28h] [rbp-38h]
  __int64 v74; // [rsp+28h] [rbp-38h]
  size_t v75; // [rsp+28h] [rbp-38h]

  result = sub_B873F0(*(_QWORD *)(a1 + 8), a2);
  if ( !*(_BYTE *)(result + 160) )
  {
    v4 = *(__int64 **)(a1 + 216);
    v5 = result;
    v6 = 2LL * *(unsigned int *)(a1 + 232);
    v7 = &v4[v6];
    if ( *(_DWORD *)(a1 + 224) )
    {
      for ( ; v4 != v7; v4 += 2 )
      {
        if ( *v4 != -8192 && *v4 != -4096 )
          break;
      }
    }
    else
    {
      v4 = (__int64 *)((char *)v4 + v6 * 8);
    }
    while ( v4 != v7 )
    {
      while ( 1 )
      {
        for ( i = v4 + 2; i != v7; i += 2 )
        {
          if ( *i != -4096 && *i != -8192 )
            break;
        }
        if ( !(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4[1] + 112LL))(v4[1]) )
        {
          v14 = *(_QWORD **)(v5 + 112);
          v15 = &v14[*(unsigned int *)(v5 + 120)];
          if ( v15 == sub_B7EAB0(v14, (__int64)v15, v4) )
            break;
        }
        v4 = i;
        if ( i == v7 )
          goto LABEL_12;
      }
      if ( (int)qword_4F81B88 > 3 )
      {
        v74 = v4[1];
        v34 = sub_C5F790(v14);
        v35 = *(_QWORD *)(v34 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v34 + 24) - v35) <= 4 )
        {
          v34 = sub_CB6200(v34, " -- '", 5);
        }
        else
        {
          *(_DWORD *)v35 = 539831584;
          *(_BYTE *)(v35 + 4) = 39;
          *(_QWORD *)(v34 + 32) += 5LL;
        }
        v70 = v34;
        v36 = (*(__int64 (__fastcall **)(__int64 *))(*a2 + 16))(a2);
        v38 = v70;
        v39 = (const char *)v36;
        v40 = *(__m128i **)(v70 + 32);
        v41 = *(_QWORD *)(v70 + 24) - (_QWORD)v40;
        if ( v37 > v41 )
        {
          v55 = sub_CB6200(v70, v39, v37);
          v40 = *(__m128i **)(v55 + 32);
          v38 = v55;
          v41 = *(_QWORD *)(v55 + 24) - (_QWORD)v40;
        }
        else if ( v37 )
        {
          v67 = v70;
          v72 = v37;
          memcpy(v40, v39, v37);
          v38 = v67;
          v57 = *(_QWORD *)(v67 + 24);
          v58 = (__m128i *)(*(_QWORD *)(v67 + 32) + v72);
          *(_QWORD *)(v67 + 32) = v58;
          v40 = v58;
          v41 = v57 - (_QWORD)v58;
        }
        if ( v41 <= 0x14 )
        {
          v39 = "' is not preserving '";
          v40 = (__m128i *)v38;
          sub_CB6200(v38, "' is not preserving '", 21);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F55300);
          v40[1].m128i_i32[0] = 543649385;
          v40[1].m128i_i8[4] = 39;
          *v40 = si128;
          *(_QWORD *)(v38 + 32) += 21LL;
        }
        v71 = sub_C5F790(v40);
        v43 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)v74 + 16LL))(v74, v39);
        v45 = v71;
        v46 = (const void *)v43;
        v47 = *(_WORD **)(v71 + 32);
        v48 = *(_QWORD *)(v71 + 24) - (_QWORD)v47;
        if ( v48 < v44 )
        {
          v56 = sub_CB6200(v71, v46, v44);
          v47 = *(_WORD **)(v56 + 32);
          v45 = v56;
          v48 = *(_QWORD *)(v56 + 24) - (_QWORD)v47;
        }
        else if ( v44 )
        {
          v75 = v44;
          memcpy(v47, v46, v44);
          v45 = v71;
          v59 = *(_QWORD *)(v71 + 24);
          v60 = (_WORD *)(*(_QWORD *)(v71 + 32) + v75);
          *(_QWORD *)(v71 + 32) = v60;
          v47 = v60;
          v48 = v59 - (_QWORD)v60;
        }
        if ( v48 <= 1 )
        {
          sub_CB6200(v45, "'\n", 2);
        }
        else
        {
          *v47 = 2599;
          *(_QWORD *)(v45 + 32) += 2LL;
        }
      }
      *v4 = -8192;
      v4 = i;
      --*(_DWORD *)(a1 + 224);
      ++*(_DWORD *)(a1 + 228);
    }
LABEL_12:
    v73 = (__int64 *)(a1 + 160);
    v69 = (__int64 *)(a1 + 208);
    do
    {
      v9 = *v73;
      if ( *v73 )
      {
        v10 = *(__int64 **)(v9 + 8);
        v11 = 2LL * *(unsigned int *)(v9 + 24);
        v12 = &v10[v11];
        if ( *(_DWORD *)(v9 + 16) )
        {
          if ( v10 != v12 )
          {
            while ( *v10 == -8192 || *v10 == -4096 )
            {
              v10 += 2;
              if ( v12 == v10 )
                goto LABEL_13;
            }
          }
        }
        else
        {
          v10 = (__int64 *)((char *)v10 + v11 * 8);
        }
        while ( v12 != v10 )
        {
          for ( j = v10 + 2; v12 != j; j += 2 )
          {
            if ( *j != -4096 && *j != -8192 )
              break;
          }
          if ( (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10[1] + 112LL))(v10[1])
            || (v16 = *(_QWORD **)(v5 + 112),
                v17 = &v16[*(unsigned int *)(v5 + 120)],
                v17 != sub_B7EAB0(v16, (__int64)v17, v10)) )
          {
            v10 = j;
          }
          else
          {
            if ( (int)qword_4F81B88 > 3 )
            {
              v65 = v10[1];
              v18 = sub_C5F790(v16);
              v19 = *(_QWORD *)(v18 + 32);
              v20 = v18;
              if ( (unsigned __int64)(*(_QWORD *)(v18 + 24) - v19) <= 4 )
              {
                v20 = sub_CB6200(v18, " -- '", 5);
              }
              else
              {
                *(_DWORD *)v19 = 539831584;
                *(_BYTE *)(v19 + 4) = 39;
                *(_QWORD *)(v18 + 32) += 5LL;
              }
              v62 = v20;
              v21 = (*(__int64 (__fastcall **)(__int64 *))(*a2 + 16))(a2);
              v23 = v62;
              v24 = (const char *)v21;
              v25 = *(__m128i **)(v62 + 32);
              v26 = *(_QWORD *)(v62 + 24) - (_QWORD)v25;
              if ( v26 < v22 )
              {
                v49 = sub_CB6200(v62, v24, v22);
                v25 = *(__m128i **)(v49 + 32);
                v23 = v49;
                v26 = *(_QWORD *)(v49 + 24) - (_QWORD)v25;
              }
              else if ( v22 )
              {
                v61 = v62;
                v64 = v22;
                memcpy(v25, v24, v22);
                v23 = v61;
                v51 = *(_QWORD *)(v61 + 24);
                v52 = (__m128i *)(*(_QWORD *)(v61 + 32) + v64);
                *(_QWORD *)(v61 + 32) = v52;
                v25 = v52;
                v26 = v51 - (_QWORD)v52;
              }
              if ( v26 <= 0x14 )
              {
                v24 = "' is not preserving '";
                v25 = (__m128i *)v23;
                sub_CB6200(v23, "' is not preserving '", 21);
              }
              else
              {
                v27 = _mm_load_si128((const __m128i *)&xmmword_3F55300);
                v25[1].m128i_i32[0] = 543649385;
                v25[1].m128i_i8[4] = 39;
                *v25 = v27;
                *(_QWORD *)(v23 + 32) += 21LL;
              }
              v63 = sub_C5F790(v25);
              v28 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)v65 + 16LL))(v65, v24);
              v30 = v63;
              v31 = (const void *)v28;
              v32 = *(_WORD **)(v63 + 32);
              v33 = *(_QWORD *)(v63 + 24) - (_QWORD)v32;
              if ( v29 > v33 )
              {
                v50 = sub_CB6200(v63, v31, v29);
                v32 = *(_WORD **)(v50 + 32);
                v30 = v50;
                v33 = *(_QWORD *)(v50 + 24) - (_QWORD)v32;
              }
              else if ( v29 )
              {
                v66 = v29;
                memcpy(v32, v31, v29);
                v30 = v63;
                v53 = *(_QWORD *)(v63 + 24);
                v54 = (_WORD *)(*(_QWORD *)(v63 + 32) + v66);
                *(_QWORD *)(v63 + 32) = v54;
                v32 = v54;
                v33 = v53 - (_QWORD)v54;
              }
              if ( v33 <= 1 )
              {
                sub_CB6200(v30, "'\n", 2);
              }
              else
              {
                *v32 = 2599;
                *(_QWORD *)(v30 + 32) += 2LL;
              }
            }
            *v10 = -8192;
            v10 = j;
            --*(_DWORD *)(v9 + 16);
            ++*(_DWORD *)(v9 + 20);
          }
        }
      }
LABEL_13:
      result = (__int64)++v73;
    }
    while ( v69 != v73 );
  }
  return result;
}
