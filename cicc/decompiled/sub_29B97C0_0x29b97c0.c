// Function: sub_29B97C0
// Address: 0x29b97c0
//
void __fastcall sub_29B97C0(char *src, char *a2)
{
  char *v2; // r8
  __int64 v4; // rbx
  double *v5; // rax
  __int64 v6; // rcx
  char *i; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  double v10; // xmm0_8
  __int64 v11; // rcx
  double v12; // xmm1_8
  double v13; // xmm2_8
  double v14; // xmm0_8
  char *v15; // r15
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  double v19; // xmm0_8
  double v20; // xmm1_8
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  double v23; // xmm1_8
  double v24; // xmm2_8
  double v25; // xmm0_8
  __int64 v26; // rdx
  __int64 v27; // rsi

  if ( src != a2 )
  {
    v2 = src + 8;
    if ( src + 8 != a2 )
    {
      do
      {
        v4 = *(_QWORD *)v2;
        v5 = *(double **)src;
        v6 = ***(_QWORD ***)(*(_QWORD *)v2 + 32LL);
        if ( (v6 == 0) == (***(_QWORD ***)(*(_QWORD *)src + 32LL) == 0) )
        {
          v18 = *((_QWORD *)v5 + 3);
          if ( v18 < 0 )
          {
            v27 = (_QWORD)v5[3] & 1LL | (*((_QWORD *)v5 + 3) >> 1);
            v19 = (double)(int)v27 + (double)(int)v27;
          }
          else
          {
            v19 = (double)(int)v18;
          }
          v20 = v5[2];
          v21 = *(_QWORD *)v5;
          v22 = *(_QWORD *)(v4 + 24);
          v23 = v20 / v19;
          if ( v22 < 0 )
          {
            v26 = *(_QWORD *)(v4 + 24) & 1LL | (*(_QWORD *)(v4 + 24) >> 1);
            v24 = (double)(int)v26 + (double)(int)v26;
          }
          else
          {
            v24 = (double)(int)v22;
          }
          v25 = *(double *)(v4 + 16) / v24;
          if ( v25 <= v23 && (v23 > v25 || v21 <= *(_QWORD *)v4) )
          {
            for ( i = v2; ; i -= 8 )
            {
LABEL_8:
              v8 = *((_QWORD *)i - 1);
              if ( (v6 == 0) == (***(_QWORD ***)(v8 + 32) == 0) )
              {
                v9 = *(_QWORD *)(v8 + 24);
                if ( v9 < 0 )
                {
                  v17 = *(_QWORD *)(v8 + 24) & 1LL | (*(_QWORD *)(v8 + 24) >> 1);
                  v10 = (double)(int)v17 + (double)(int)v17;
                }
                else
                {
                  v10 = (double)(int)v9;
                }
                v11 = *(_QWORD *)(v4 + 24);
                v12 = *(double *)(v8 + 16) / v10;
                if ( v11 < 0 )
                {
                  v16 = *(_QWORD *)(v4 + 24) & 1LL | (*(_QWORD *)(v4 + 24) >> 1);
                  v13 = (double)(int)v16 + (double)(int)v16;
                }
                else
                {
                  v13 = (double)(int)v11;
                }
                v14 = *(double *)(v4 + 16) / v13;
                if ( v14 <= v12 && (v12 > v14 || *(_QWORD *)v8 <= *(_QWORD *)v4) )
                {
LABEL_16:
                  *(_QWORD *)i = v4;
                  v15 = v2 + 8;
                  goto LABEL_17;
                }
              }
              else if ( v6 )
              {
                goto LABEL_16;
              }
              *(_QWORD *)i = v8;
              v6 = ***(_QWORD ***)(v4 + 32);
            }
          }
        }
        else
        {
          i = v2;
          if ( v6 )
            goto LABEL_8;
        }
        v15 = v2 + 8;
        if ( src != v2 )
          memmove(src + 8, src, v2 - src);
        *(_QWORD *)src = v4;
LABEL_17:
        v2 = v15;
      }
      while ( a2 != v15 );
    }
  }
}
