// Function: sub_892C00
// Address: 0x892c00
//
_QWORD *__fastcall sub_892C00(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  const __m128i *v3; // r15
  _QWORD *m128i_i64; // r13
  _QWORD *v5; // r12
  _QWORD *v6; // rdx
  _QWORD *v7; // r14
  __int64 v8; // rax
  __m128i *v9; // rbx
  _DWORD *v10; // rax
  _QWORD *v11; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rax
  int v16; // [rsp+Ch] [rbp-34h]

  if ( a1 )
  {
    v2 = sub_892BC0(a2[2]);
    v3 = *(const __m128i **)(a1 + 24);
    v16 = *(_DWORD *)(v2 + 4);
    if ( v3 )
    {
      m128i_i64 = 0;
      v5 = 0;
      while ( 1 )
      {
        v10 = (_DWORD *)v3[4].m128i_i64[0];
        if ( v10[1] < v16 )
        {
          v9 = sub_8665B0(v3);
          if ( !v5 )
            goto LABEL_11;
LABEL_7:
          *m128i_i64 = v9;
          m128i_i64 = v9->m128i_i64;
          goto LABEL_8;
        }
        v6 = (_QWORD *)a2[7];
        if ( v6 )
        {
          v7 = *(_QWORD **)(*v6 + 8LL * (unsigned int)(*v10 - 1));
          if ( v7 )
            goto LABEL_6;
LABEL_8:
          v3 = (const __m128i *)v3->m128i_i64[0];
          if ( !v3 )
            return v5;
        }
        else
        {
          v13 = (_QWORD *)a2[2];
          if ( v13 )
          {
            v7 = (_QWORD *)a2[4];
            while ( *(_DWORD *)(v3->m128i_i64[1] + 56) != *(_DWORD *)(v7[1] + 56LL) )
            {
              v13 = (_QWORD *)*v13;
              v7 = (_QWORD *)*v7;
              if ( !v13 )
                goto LABEL_17;
            }
          }
          else
          {
            v7 = (_QWORD *)a2[4];
LABEL_17:
            v14 = (_QWORD *)a2[3];
            if ( v14 )
            {
              while ( *(_DWORD *)(v3->m128i_i64[1] + 56) != *(_DWORD *)(v7[1] + 56LL) )
              {
                v14 = (_QWORD *)*v14;
                v7 = (_QWORD *)*v7;
                if ( !v14 )
                  goto LABEL_20;
              }
            }
            else
            {
LABEL_20:
              if ( !a2[4] )
              {
LABEL_24:
                v15 = sub_866270(0);
                *(_QWORD *)(v15 + 72) = 0;
                *(_QWORD *)(v15 + 64) = sub_892BC0(0);
                BUG();
              }
              v7 = (_QWORD *)a2[4];
              while ( v3->m128i_i64[1] != v7[1] )
              {
                v7 = (_QWORD *)*v7;
                if ( !v7 )
                  goto LABEL_24;
              }
            }
          }
LABEL_6:
          v8 = sub_866270(0);
          *(_QWORD *)(v8 + 72) = v7;
          v9 = (__m128i *)v8;
          *(_QWORD *)(v8 + 64) = sub_892BC0((__int64)v7);
          v9->m128i_i64[1] = v7[1];
          if ( v5 )
            goto LABEL_7;
LABEL_11:
          v11 = sub_8663A0();
          m128i_i64 = v9->m128i_i64;
          v11[3] = v9;
          v3 = (const __m128i *)v3->m128i_i64[0];
          v5 = v11;
          if ( !v3 )
            return v5;
        }
      }
    }
  }
  return 0;
}
