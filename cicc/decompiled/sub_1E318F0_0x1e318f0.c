// Function: sub_1E318F0
// Address: 0x1e318f0
//
void __fastcall sub_1E318F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 (*v5)(); // rax
  __int64 v6; // r9
  __int64 (*v7)(); // rax
  char *v8; // rsi
  int v9; // r14d
  bool v10; // r15
  int v11; // r13d
  int v12; // ebx
  void *v13; // rdx
  __int64 (*v14)(); // rax
  __m128i *v15; // r8
  __int64 v16; // r14
  __m128i *v17; // rax
  __m128i v18; // xmm0
  __m128i *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rcx
  char *v24; // rsi
  size_t v25; // rax
  size_t v26; // r15
  int v27; // eax
  __int64 (*v28)(); // rax
  __m128i *v29; // rdx
  __m128i si128; // xmm0
  size_t v31; // rdx
  char *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  size_t v36; // rax
  size_t v37; // rbx
  void *v38; // rdx
  __int64 v39; // rax
  _WORD *v40; // rdx
  __m128i *v41; // [rsp-50h] [rbp-50h]
  __int64 v42; // [rsp-48h] [rbp-48h]
  __int64 v43; // [rsp-40h] [rbp-40h]
  __int64 v44; // [rsp-40h] [rbp-40h]
  __int64 v45; // [rsp-40h] [rbp-40h]
  __int64 v46; // [rsp-40h] [rbp-40h]
  __int64 v47; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)a2 )
  {
    if ( (*(_DWORD *)a2 & 0xFFF00) != 0 )
    {
      v2 = *(_QWORD *)(a2 + 16);
      if ( v2 )
      {
        v3 = *(_QWORD *)(v2 + 24);
        if ( v3 )
        {
          v4 = *(_QWORD *)(v3 + 56);
          if ( v4 )
          {
            v5 = *(__int64 (**)())(**(_QWORD **)(v4 + 16) + 40LL);
            if ( v5 == sub_1D00B00 )
              BUG();
            v6 = v5();
            v7 = *(__int64 (**)())(*(_QWORD *)v6 + 976LL);
            if ( *(_BYTE *)a2 )
              v8 = (char *)((*(_DWORD *)a2 >> 8) & 0xFFF);
            else
              v8 = 0;
            if ( v7 == sub_1E308E0 )
            {
              v9 = 0;
              v10 = 0;
              v11 = 0;
              v12 = 0;
            }
            else
            {
              v45 = v6;
              v39 = ((__int64 (__fastcall *)(__int64, char *))v7)(v6, v8);
              v6 = v45;
              v12 = v39;
              v11 = HIDWORD(v39);
              v10 = (_DWORD)v39 != 0;
              v9 = HIDWORD(v39) | v39;
            }
            v13 = *(void **)(a1 + 24);
            if ( *(_QWORD *)(a1 + 16) - (_QWORD)v13 <= 0xCu )
            {
              v8 = "target-flags(";
              v43 = v6;
              sub_16E7EE0(a1, "target-flags(", 0xDu);
              v6 = v43;
            }
            else
            {
              qmemcpy(v13, "target-flags(", 13);
              *(_QWORD *)(a1 + 24) += 13LL;
            }
            if ( !v9 )
            {
              v38 = *(void **)(a1 + 24);
              if ( *(_QWORD *)(a1 + 16) - (_QWORD)v38 > 0xAu )
              {
                qmemcpy(v38, "<unknown>) ", 11);
                *(_QWORD *)(a1 + 24) += 11LL;
                return;
              }
              v31 = 11;
              v32 = "<unknown>) ";
LABEL_49:
              sub_16E7EE0(a1, v32, v31);
              return;
            }
            if ( v12 )
            {
              v28 = *(__int64 (**)())(*(_QWORD *)v6 + 984LL);
              if ( v28 == sub_1E308F0 )
                goto LABEL_46;
              v44 = v6;
              v33 = ((__int64 (__fastcall *)(__int64, char *))v28)(v6, v8);
              v6 = v44;
              v35 = v33 + 16 * v34;
              if ( v35 == v33 )
                goto LABEL_46;
              while ( v12 != *(_DWORD *)v33 )
              {
                v33 += 16;
                if ( v35 == v33 )
                  goto LABEL_46;
              }
              v8 = *(char **)(v33 + 8);
              if ( !v8 )
              {
LABEL_46:
                v29 = *(__m128i **)(a1 + 24);
                if ( *(_QWORD *)(a1 + 16) - (_QWORD)v29 <= 0x14u )
                {
                  v8 = "<unknown target flag>";
                  v47 = v6;
                  sub_16E7EE0(a1, "<unknown target flag>", 0x15u);
                  v6 = v47;
                }
                else
                {
                  si128 = _mm_load_si128((const __m128i *)&xmmword_42EB980);
                  v29[1].m128i_i32[0] = 1734437990;
                  v29[1].m128i_i8[4] = 62;
                  *v29 = si128;
                  *(_QWORD *)(a1 + 24) += 21LL;
                }
              }
              else
              {
                v36 = strlen(*(const char **)(v33 + 8));
                v6 = v44;
                v37 = v36;
                if ( v36 > *(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 24) )
                {
                  sub_16E7EE0(a1, v8, v36);
                  v6 = v44;
                }
                else if ( v36 )
                {
                  memcpy(*(void **)(a1 + 24), v8, v36);
                  *(_QWORD *)(a1 + 24) += v37;
                  v6 = v44;
                }
              }
            }
            if ( !v11 )
            {
              v40 = *(_WORD **)(a1 + 24);
              if ( *(_QWORD *)(a1 + 16) - (_QWORD)v40 > 1u )
              {
                *v40 = 8233;
                *(_QWORD *)(a1 + 24) += 2LL;
                return;
              }
LABEL_48:
              v31 = 2;
              v32 = ") ";
              goto LABEL_49;
            }
            v14 = *(__int64 (**)())(*(_QWORD *)v6 + 992LL);
            if ( v14 == sub_1E30900 )
            {
              v15 = *(__m128i **)(a1 + 24);
              v16 = *(_QWORD *)(a1 + 16);
            }
            else
            {
              v20 = ((__int64 (__fastcall *)(__int64, char *))v14)(v6, v8);
              v15 = *(__m128i **)(a1 + 24);
              v16 = *(_QWORD *)(a1 + 16);
              v21 = v20;
              v22 *= 16;
              v23 = v20 + v22;
              if ( v20 + v22 != v20 )
              {
                do
                {
                  while ( *(_DWORD *)v21 != (v11 & *(_DWORD *)v21) )
                  {
                    v21 += 16;
                    if ( v23 == v21 )
                      goto LABEL_33;
                  }
                  if ( v10 )
                  {
                    if ( (unsigned __int64)(v16 - (_QWORD)v15) <= 1 )
                    {
                      v46 = v23;
                      sub_16E7EE0(a1, ", ", 2u);
                      v15 = *(__m128i **)(a1 + 24);
                      v16 = *(_QWORD *)(a1 + 16);
                      v23 = v46;
                    }
                    else
                    {
                      v15->m128i_i16[0] = 8236;
                      v16 = *(_QWORD *)(a1 + 16);
                      *(_QWORD *)(a1 + 24) += 2LL;
                      v15 = *(__m128i **)(a1 + 24);
                    }
                  }
                  v24 = *(char **)(v21 + 8);
                  if ( v24 )
                  {
                    v41 = v15;
                    v42 = v23;
                    v25 = strlen(*(const char **)(v21 + 8));
                    v15 = v41;
                    v26 = v25;
                    v23 = v42;
                    if ( v25 > v16 - (__int64)v41 )
                    {
                      sub_16E7EE0(a1, v24, v25);
                      v15 = *(__m128i **)(a1 + 24);
                      v16 = *(_QWORD *)(a1 + 16);
                      v23 = v42;
                    }
                    else if ( v25 )
                    {
                      memcpy(v41, v24, v25);
                      *(_QWORD *)(a1 + 24) += v26;
                      v16 = *(_QWORD *)(a1 + 16);
                      v15 = *(__m128i **)(a1 + 24);
                      v23 = v42;
                    }
                  }
                  v27 = *(_DWORD *)v21;
                  v21 += 16;
                  v10 = 1;
                  v11 &= ~v27;
                }
                while ( v23 != v21 );
LABEL_33:
                v19 = v15;
                if ( !v11 )
                {
LABEL_34:
                  if ( (unsigned __int64)(v16 - (_QWORD)v19) > 1 )
                  {
                    v19->m128i_i16[0] = 8233;
                    *(_QWORD *)(a1 + 24) += 2LL;
                    return;
                  }
                  goto LABEL_48;
                }
              }
            }
            v17 = v15;
            if ( v10 )
            {
              if ( (unsigned __int64)(v16 - (_QWORD)v15) <= 1 )
              {
                sub_16E7EE0(a1, ", ", 2u);
                v17 = *(__m128i **)(a1 + 24);
                v16 = *(_QWORD *)(a1 + 16);
              }
              else
              {
                v15->m128i_i16[0] = 8236;
                v16 = *(_QWORD *)(a1 + 16);
                v17 = (__m128i *)(*(_QWORD *)(a1 + 24) + 2LL);
                *(_QWORD *)(a1 + 24) = v17;
              }
            }
            if ( (unsigned __int64)(v16 - (_QWORD)v17) <= 0x1C )
            {
              sub_16E7EE0(a1, "<unknown bitmask target flag>", 0x1Du);
              v19 = *(__m128i **)(a1 + 24);
              v16 = *(_QWORD *)(a1 + 16);
            }
            else
            {
              v18 = _mm_load_si128((const __m128i *)&xmmword_42EB990);
              qmemcpy(&v17[1], " target flag>", 13);
              *v17 = v18;
              v16 = *(_QWORD *)(a1 + 16);
              v19 = (__m128i *)(*(_QWORD *)(a1 + 24) + 29LL);
              *(_QWORD *)(a1 + 24) = v19;
            }
            goto LABEL_34;
          }
        }
      }
    }
  }
}
