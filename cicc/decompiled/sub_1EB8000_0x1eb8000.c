// Function: sub_1EB8000
// Address: 0x1eb8000
//
__int64 __fastcall sub_1EB8000(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v6; // r15d
  unsigned __int16 v7; // r14
  __int16 v8; // r13
  int v10; // ebx
  __int64 result; // rax
  int v12; // ecx
  __int64 v13; // rdx
  unsigned int v14; // ecx
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned int v17; // eax
  __m128i *v18; // r9
  _QWORD *v19; // rsi
  __int64 v20; // rcx
  unsigned int v21; // edx
  __int16 v22; // r14
  _WORD *v23; // rdx
  _WORD *v24; // rdi
  unsigned __int16 v25; // dx
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int16 v29; // r15
  __int16 v30; // dx
  int *v31; // rax
  int v32; // esi
  __int16 *v33; // rsi
  __int16 v34; // ax
  _WORD *v35; // rsi
  __int16 v36; // dx
  __int16 v37; // cx
  unsigned int v38; // esi
  __int64 v39; // rdi
  __int64 v40; // r10
  unsigned int v41; // eax
  __m128i *v42; // r11
  __int64 v44; // [rsp+18h] [rbp-78h]
  int v45; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v46; // [rsp+28h] [rbp-68h]
  char v47; // [rsp+30h] [rbp-60h]
  unsigned __int16 v48; // [rsp+38h] [rbp-58h]
  _WORD *v49; // [rsp+40h] [rbp-50h]
  int v50; // [rsp+48h] [rbp-48h]
  unsigned __int16 v51; // [rsp+50h] [rbp-40h]
  __int64 v52; // [rsp+58h] [rbp-38h]

  v6 = (unsigned __int16)a3;
  v7 = a3;
  v8 = a3;
  v10 = a4;
  sub_1EB6840(a1, a3, a3, a4, a5, a6);
  result = 4LL * v7 + *(_QWORD *)(a1 + 648);
  v12 = *(_DWORD *)result;
  if ( *(_DWORD *)result )
  {
    if ( (unsigned int)(v12 - 1) > 1 )
    {
      v13 = *(_QWORD *)(a1 + 600);
      v14 = v12 & 0x7FFFFFFF;
      v15 = *(unsigned int *)(a1 + 400);
      v16 = *(_QWORD *)(a1 + 392);
      v17 = *(unsigned __int8 *)(v13 + v14);
      if ( v17 < (unsigned int)v15 )
      {
        while ( 1 )
        {
          v18 = (__m128i *)(v16 + 24LL * v17);
          if ( v14 == (v18->m128i_i32[2] & 0x7FFFFFFF) )
            break;
          v17 += 256;
          if ( (unsigned int)v15 <= v17 )
            goto LABEL_39;
        }
      }
      else
      {
LABEL_39:
        v18 = (__m128i *)(v16 + 24 * v15);
      }
      sub_1EB7C50(a1, a2, v18);
      result = 4LL * v7 + *(_QWORD *)(a1 + 648);
    }
    *(_DWORD *)result = v10;
  }
  else
  {
    *(_DWORD *)result = v10;
    v19 = *(_QWORD **)(a1 + 248);
    v45 = v6;
    if ( !v19 )
    {
      v46 = 0;
      v47 = 0;
      v48 = 0;
      v49 = 0;
      v50 = 0;
      v51 = 0;
      v52 = 0;
      BUG();
    }
    v47 = 0;
    v46 = v19 + 1;
    v48 = 0;
    v49 = 0;
    v52 = 0;
    v20 = v19[7];
    v50 = 0;
    v51 = 0;
    v44 = 24LL * v7;
    v21 = *(_DWORD *)(v19[1] + v44 + 16);
    result = v21 & 0xF;
    v22 = (v21 & 0xF) * v7;
    v23 = (_WORD *)(v20 + 2LL * (v21 >> 4));
    v24 = v23 + 1;
    v48 = *v23 + v22;
    v49 = v23 + 1;
    do
    {
      if ( !v24 )
        break;
      v50 = *(_DWORD *)(v19[6] + 4LL * v48);
      v25 = v50;
      if ( (_WORD)v50 )
      {
        while ( 2 )
        {
          v26 = *(unsigned int *)(v19[1] + 24LL * v25 + 8);
          v27 = v19[7];
          v51 = v25;
          v28 = v27 + 2 * v26;
          v52 = v28;
          while ( v28 )
          {
            v29 = v51;
            if ( v8 != v51 )
            {
              while ( 1 )
              {
                v31 = (int *)(4LL * v29 + *(_QWORD *)(a1 + 648));
                v32 = *v31;
                if ( *v31 )
                {
                  if ( (unsigned int)(v32 - 1) > 1 )
                  {
                    v38 = v32 & 0x7FFFFFFF;
                    v39 = *(unsigned int *)(a1 + 400);
                    v40 = *(_QWORD *)(a1 + 392);
                    v41 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 600) + v38);
                    if ( v41 < (unsigned int)v39 )
                    {
                      while ( 1 )
                      {
                        v42 = (__m128i *)(v40 + 24LL * v41);
                        if ( v38 == (v42->m128i_i32[2] & 0x7FFFFFFF) )
                          break;
                        v41 += 256;
                        if ( (unsigned int)v39 <= v41 )
                          goto LABEL_38;
                      }
                    }
                    else
                    {
LABEL_38:
                      v42 = (__m128i *)(v40 + 24 * v39);
                    }
                    sub_1EB7C50(a1, a2, v42);
                    v31 = (int *)(4LL * v29 + *(_QWORD *)(a1 + 648));
                  }
                  *v31 = 0;
                  v33 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 56LL)
                                  + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 8LL) + v44 + 8));
                  v34 = *v33;
                  v35 = v33 + 1;
                  v36 = v8 + v34;
                  if ( !v34 )
                    v35 = 0;
LABEL_29:
                  result = (__int64)v35;
                  while ( result )
                  {
                    if ( v36 == v29 )
                      return result;
                    v37 = *(_WORD *)result;
                    v35 = 0;
                    result += 2;
                    v36 += v37;
                    if ( !v37 )
                      goto LABEL_29;
                  }
                }
                result = sub_1E1D5E0((__int64)&v45);
                if ( !v49 )
                  return result;
                v29 = v51;
              }
            }
            v28 += 2;
            v52 = v28;
            v30 = *(_WORD *)(v28 - 2);
            v51 = v8 + v30;
            if ( !v30 )
            {
              v52 = 0;
              break;
            }
          }
          v25 = HIWORD(v50);
          v50 = HIWORD(v50);
          if ( v25 )
            continue;
          break;
        }
      }
      v49 = ++v24;
      result = (unsigned __int16)*(v24 - 1);
      v48 += result;
    }
    while ( (_WORD)result );
  }
  return result;
}
