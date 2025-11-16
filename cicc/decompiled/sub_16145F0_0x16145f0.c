// Function: sub_16145F0
// Address: 0x16145f0
//
__int64 __fastcall sub_16145F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 *v7; // r15
  __int64 *i; // rbx
  __int64 *v9; // rbx
  __int64 v10; // r8
  __int64 *v11; // r13
  __int64 v12; // r8
  __int64 *v13; // r14
  __int64 *j; // r15
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r10
  __int64 v24; // rax
  size_t v25; // rdx
  __int64 v26; // r10
  const char *v27; // rsi
  _OWORD *v28; // rdi
  unsigned __int64 v29; // rax
  __m128i si128; // xmm0
  const char *v31; // rax
  size_t v32; // rdx
  __int64 v33; // r10
  void *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  const char *v37; // rax
  size_t v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdx
  const char *v41; // rax
  size_t v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-58h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  __int64 v47; // [rsp+10h] [rbp-50h]
  size_t v48; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+18h] [rbp-48h]
  __int64 v50; // [rsp+18h] [rbp-48h]
  size_t v51; // [rsp+18h] [rbp-48h]
  __int64 *v53; // [rsp+28h] [rbp-38h]
  __int64 v54; // [rsp+28h] [rbp-38h]
  __int64 v55; // [rsp+28h] [rbp-38h]

  result = sub_16135E0(*(_QWORD *)(a1 + 16), a2);
  if ( !*(_BYTE *)(result + 160) )
  {
    v4 = *(__int64 **)(a1 + 232);
    v5 = result;
    v6 = 2LL * *(unsigned int *)(a1 + 248);
    v7 = &v4[v6];
    if ( *(_DWORD *)(a1 + 240) )
    {
      for ( ; v7 != v4; v4 += 2 )
      {
        if ( *v4 != -8 && *v4 != -4 )
          break;
      }
    }
    else
    {
      v4 = (__int64 *)((char *)v4 + v6 * 8);
    }
    while ( v7 != v4 )
    {
      while ( 1 )
      {
        for ( i = v4 + 2; v7 != i; i += 2 )
        {
          if ( *i != -4 && *i != -8 )
            break;
        }
        if ( !(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4[1] + 112LL))(v4[1]) )
        {
          v15 = *(_QWORD **)(v5 + 112);
          v16 = &v15[*(unsigned int *)(v5 + 120)];
          if ( v16 == sub_160D180(v15, (__int64)v16, v4) )
            break;
        }
        v4 = i;
        if ( v7 == i )
          goto LABEL_12;
      }
      if ( dword_4F9EB40 > 3 )
      {
        v50 = v4[1];
        v36 = sub_16BA580(v15, v16, v17);
        v54 = sub_1263B40(v36, " -- '");
        v37 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
        v39 = sub_1549FF0(v54, v37, v38);
        sub_1263B40(v39, "' is not preserving '");
        v55 = sub_16BA580(v39, "' is not preserving '", v40);
        v41 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v50 + 16LL))(v50);
        v43 = sub_1549FF0(v55, v41, v42);
        sub_1263B40(v43, "'\n");
      }
      *v4 = -8;
      v4 = i;
      --*(_DWORD *)(a1 + 240);
      ++*(_DWORD *)(a1 + 244);
    }
LABEL_12:
    v9 = (__int64 *)(a1 + 168);
    v53 = (__int64 *)(a1 + 224);
    do
    {
      result = *v9;
      if ( *v9 )
      {
        v10 = *(unsigned int *)(result + 24);
        v11 = *(__int64 **)(result + 8);
        result = *(unsigned int *)(result + 16);
        v12 = 2 * v10;
        v13 = &v11[v12];
        if ( !(_DWORD)result )
        {
          v11 = (__int64 *)((char *)v11 + v12 * 8);
LABEL_17:
          while ( v11 != v13 )
          {
            for ( j = v11 + 2; j != v13; j += 2 )
            {
              if ( *j != -4 && *j != -8 )
                break;
            }
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11[1] + 112LL))(v11[1]);
            if ( result
              || (v18 = *(_QWORD **)(v5 + 112),
                  v19 = (__int64)&v18[*(unsigned int *)(v5 + 120)],
                  result = (__int64)sub_160D180(v18, v19, v11),
                  v19 != result) )
            {
              v11 = j;
            }
            else
            {
              if ( dword_4F9EB40 > 3 )
              {
                v49 = v11[1];
                v21 = sub_16BA580(v18, v19, v20);
                v22 = *(_QWORD *)(v21 + 24);
                v23 = v21;
                if ( (unsigned __int64)(*(_QWORD *)(v21 + 16) - v22) <= 4 )
                {
                  v23 = sub_16E7EE0(v21, " -- '", 5);
                }
                else
                {
                  *(_DWORD *)v22 = 539831584;
                  *(_BYTE *)(v22 + 4) = 39;
                  *(_QWORD *)(v21 + 24) += 5LL;
                }
                v46 = v23;
                v24 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
                v26 = v46;
                v27 = (const char *)v24;
                v28 = *(_OWORD **)(v46 + 24);
                v29 = *(_QWORD *)(v46 + 16) - (_QWORD)v28;
                if ( v25 > v29 )
                {
                  v35 = sub_16E7EE0(v46, v27);
                  v28 = *(_OWORD **)(v35 + 24);
                  v26 = v35;
                  v29 = *(_QWORD *)(v35 + 16) - (_QWORD)v28;
                }
                else if ( v25 )
                {
                  v45 = v46;
                  v48 = v25;
                  memcpy(v28, v27, v25);
                  v26 = v45;
                  v44 = *(_QWORD *)(v45 + 16);
                  v25 = *(_QWORD *)(v45 + 24) + v48;
                  *(_QWORD *)(v45 + 24) = v25;
                  v28 = (_OWORD *)v25;
                  v29 = v44 - v25;
                }
                if ( v29 <= 0x14 )
                {
                  v27 = "' is not preserving '";
                  v28 = (_OWORD *)v26;
                  sub_16E7EE0(v26, "' is not preserving '", 21);
                }
                else
                {
                  si128 = _mm_load_si128((const __m128i *)&xmmword_3F55300);
                  *((_DWORD *)v28 + 4) = 543649385;
                  *((_BYTE *)v28 + 20) = 39;
                  *v28 = si128;
                  *(_QWORD *)(v26 + 24) += 21LL;
                }
                v47 = sub_16BA580(v28, v27, v25);
                v31 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v49 + 16LL))(v49);
                v33 = v47;
                v34 = *(void **)(v47 + 24);
                if ( v32 > *(_QWORD *)(v47 + 16) - (_QWORD)v34 )
                {
                  v33 = sub_16E7EE0(v47, v31);
                }
                else if ( v32 )
                {
                  v51 = v32;
                  memcpy(v34, v31, v32);
                  v33 = v47;
                  *(_QWORD *)(v47 + 24) += v51;
                }
                sub_1263B40(v33, "'\n");
              }
              result = *v9;
              *v11 = -8;
              v11 = j;
              --*(_DWORD *)(result + 16);
              ++*(_DWORD *)(result + 20);
            }
          }
          goto LABEL_13;
        }
        for ( ; v11 != v13; v11 += 2 )
        {
          result = *v11;
          if ( *v11 != -8 && result != -4 )
            goto LABEL_17;
        }
      }
LABEL_13:
      ++v9;
    }
    while ( v53 != v9 );
  }
  return result;
}
