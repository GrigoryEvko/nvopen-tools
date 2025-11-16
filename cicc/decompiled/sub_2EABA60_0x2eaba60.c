// Function: sub_2EABA60
// Address: 0x2eaba60
//
void __fastcall sub_2EABA60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 (*v6)(); // rax
  __int64 v7; // r14
  __int64 (*v8)(); // rax
  __int64 v9; // rsi
  int v10; // ecx
  bool v11; // r15
  int v12; // r13d
  int v13; // ebx
  void *v14; // rdx
  const char *v15; // rsi
  __int64 (*v16)(); // rax
  __m128i *v17; // rdx
  __m128i si128; // xmm0
  _WORD *v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // r14
  size_t v23; // rax
  void *v24; // rdi
  size_t v25; // r15
  int v26; // eax
  _WORD *v27; // rdx
  __int64 (*v28)(); // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rax
  char *v33; // [rsp-40h] [rbp-40h]
  int v34; // [rsp-40h] [rbp-40h]

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
          v4 = *(_QWORD *)(v3 + 32);
          if ( v4 )
          {
            v6 = *(__int64 (**)())(**(_QWORD **)(v4 + 16) + 128LL);
            if ( v6 == sub_2DAC790 )
              BUG();
            v7 = v6();
            v8 = *(__int64 (**)())(*(_QWORD *)v7 + 1296LL);
            if ( *(_BYTE *)a2 )
              v9 = (*(_DWORD *)a2 >> 8) & 0xFFF;
            else
              v9 = 0;
            if ( v8 == sub_2EAACD0 )
            {
              v10 = 0;
              v11 = 0;
              v12 = 0;
              v13 = 0;
            }
            else
            {
              v32 = ((__int64 (__fastcall *)(__int64, __int64))v8)(v7, v9);
              v13 = v32;
              v12 = HIDWORD(v32);
              v11 = (_DWORD)v32 != 0;
              v10 = HIDWORD(v32) | v32;
            }
            v14 = *(void **)(a1 + 32);
            if ( *(_QWORD *)(a1 + 24) - (_QWORD)v14 <= 0xCu )
            {
              v34 = v10;
              sub_CB6200(a1, "target-flags(", 0xDu);
              v10 = v34;
            }
            else
            {
              qmemcpy(v14, "target-flags(", 13);
              *(_QWORD *)(a1 + 32) += 13LL;
            }
            v15 = "<unknown>) ";
            if ( v10 )
            {
              if ( v13 )
              {
                v28 = *(__int64 (**)())(*(_QWORD *)v7 + 1304LL);
                if ( v28 == sub_2EAACE0 )
                  goto LABEL_46;
                v29 = ((__int64 (__fastcall *)(__int64, const char *))v28)(v7, "<unknown>) ");
                v31 = v29 + 16 * v30;
                if ( v31 == v29 )
                  goto LABEL_46;
                while ( v13 != *(_DWORD *)v29 )
                {
                  v29 += 16;
                  if ( v31 == v29 )
                    goto LABEL_46;
                }
                v15 = *(const char **)(v29 + 8);
                if ( !v15 )
LABEL_46:
                  v15 = "<unknown target flag>";
                sub_904010(a1, v15);
              }
              if ( v12 )
              {
                v16 = *(__int64 (**)())(*(_QWORD *)v7 + 1312LL);
                if ( v16 == sub_2EAACF0 )
                  goto LABEL_19;
                v20 = ((__int64 (__fastcall *)(__int64, const char *))v16)(v7, v15);
                v21 *= 16;
                v22 = v20 + v21;
                if ( v20 + v21 == v20 )
                  goto LABEL_19;
                do
                {
                  while ( *(_DWORD *)v20 != (v12 & *(_DWORD *)v20) )
                  {
                    v20 += 16;
                    if ( v22 == v20 )
                      goto LABEL_33;
                  }
                  if ( v11 )
                  {
                    v27 = *(_WORD **)(a1 + 32);
                    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v27 <= 1u )
                    {
                      sub_CB6200(a1, (unsigned __int8 *)", ", 2u);
                    }
                    else
                    {
                      *v27 = 8236;
                      *(_QWORD *)(a1 + 32) += 2LL;
                    }
                  }
                  if ( *(_QWORD *)(v20 + 8) )
                  {
                    v33 = *(char **)(v20 + 8);
                    v23 = strlen(v33);
                    v24 = *(void **)(a1 + 32);
                    v25 = v23;
                    if ( v23 > *(_QWORD *)(a1 + 24) - (_QWORD)v24 )
                    {
                      sub_CB6200(a1, (unsigned __int8 *)v33, v23);
                    }
                    else if ( v23 )
                    {
                      memcpy(v24, v33, v23);
                      *(_QWORD *)(a1 + 32) += v25;
                    }
                  }
                  v26 = *(_DWORD *)v20;
                  v20 += 16;
                  v11 = 1;
                  v12 &= ~v26;
                }
                while ( v22 != v20 );
LABEL_33:
                if ( v12 )
                {
LABEL_19:
                  if ( v11 )
                    sub_904010(a1, ", ");
                  v17 = *(__m128i **)(a1 + 32);
                  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v17 > 0x1Cu )
                  {
                    si128 = _mm_load_si128((const __m128i *)&xmmword_42EB990);
                    qmemcpy(&v17[1], " target flag>", 13);
                    *v17 = si128;
                    v19 = (_WORD *)(*(_QWORD *)(a1 + 32) + 29LL);
                    *(_QWORD *)(a1 + 32) = v19;
                    goto LABEL_35;
                  }
                  sub_CB6200(a1, "<unknown bitmask target flag>", 0x1Du);
                }
                v19 = *(_WORD **)(a1 + 32);
LABEL_35:
                if ( *(_QWORD *)(a1 + 24) - (_QWORD)v19 <= 1u )
                {
                  sub_CB6200(a1, (unsigned __int8 *)") ", 2u);
                }
                else
                {
                  *v19 = 8233;
                  *(_QWORD *)(a1 + 32) += 2LL;
                }
                return;
              }
              v15 = ") ";
            }
            sub_904010(a1, v15);
          }
        }
      }
    }
  }
}
