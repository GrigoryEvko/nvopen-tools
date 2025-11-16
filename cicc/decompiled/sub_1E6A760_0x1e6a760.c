// Function: sub_1E6A760
// Address: 0x1e6a760
//
__int64 __fastcall sub_1E6A760(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int v5; // r12d
  __int64 v7; // r15
  unsigned int *v8; // rdx
  int v9; // r14d
  bool v10; // sf
  __int64 v11; // rax
  unsigned int *v12; // rcx
  char *v13; // rax
  __int64 v14; // rdx
  int v15; // ecx
  char *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int32 v20; // edx
  __int64 *v22; // [rsp+20h] [rbp-90h]
  __int64 v23; // [rsp+28h] [rbp-88h]
  __int64 v24; // [rsp+30h] [rbp-80h]
  int v25; // [rsp+3Ch] [rbp-74h]
  __int64 v26; // [rsp+48h] [rbp-68h] BYREF
  __m128i v27; // [rsp+50h] [rbp-60h] BYREF
  __int64 v28; // [rsp+60h] [rbp-50h]
  __int64 v29; // [rsp+68h] [rbp-48h]
  __int64 v30; // [rsp+70h] [rbp-40h]

  result = a1[45];
  v25 = (a1[46] - result) >> 3;
  if ( v25 )
  {
    v5 = 0;
    v7 = 0;
    v8 = (unsigned int *)a1[45];
    v9 = *(_DWORD *)(result + 4);
    v10 = v9 < 0;
    if ( v9 )
      goto LABEL_3;
    while ( 1 )
    {
      result = *v8;
LABEL_16:
      v27.m128i_i16[0] = result;
      v16 = *(char **)(a2 + 160);
      v27.m128i_i32[1] = -1;
      if ( v16 == *(char **)(a2 + 168) )
      {
        result = sub_1D4B220((char **)(a2 + 152), v16, &v27);
      }
      else
      {
        if ( v16 )
        {
          result = v27.m128i_i64[0];
          *(_QWORD *)v16 = v27.m128i_i64[0];
          v16 = *(char **)(a2 + 160);
        }
        *(_QWORD *)(a2 + 160) = v16 + 8;
      }
      if ( v25 == ++v5 )
        break;
      while ( 1 )
      {
        v7 = v5;
        v8 = (unsigned int *)(a1[45] + 8LL * v5);
        v9 = v8[1];
        v10 = v9 < 0;
        if ( !v9 )
          break;
LABEL_3:
        if ( v10 )
          v11 = *(_QWORD *)(a1[3] + 16LL * (v9 & 0x7FFFFFFF) + 8);
        else
          v11 = *(_QWORD *)(a1[34] + 8LL * (unsigned int)v9);
        while ( v11 )
        {
          if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 && (*(_BYTE *)(v11 + 4) & 8) == 0 )
          {
            v22 = *(__int64 **)(a2 + 32);
            v17 = *(_QWORD *)(a4 + 8);
            v23 = *(_QWORD *)(a2 + 56);
            v26 = 0;
            v24 = (__int64)sub_1E0B640(v23, v17 + 960, &v26, 0);
            sub_1DD5BA0((__int64 *)(a2 + 16), v24);
            v18 = *v22;
            v19 = *(_QWORD *)v24;
            *(_QWORD *)(v24 + 8) = v22;
            v18 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v24 = v18 | v19 & 7;
            *(_QWORD *)(v18 + 8) = v24;
            *v22 = v24 | *v22 & 7;
            v27.m128i_i32[2] = v9;
            v27.m128i_i64[0] = 0x10000000;
            v28 = 0;
            v29 = 0;
            v30 = 0;
            sub_1E1A9C0(v24, v23, &v27);
            v20 = *(_DWORD *)(a1[45] + 8 * v7);
            v27.m128i_i64[0] = 0;
            v28 = 0;
            v27.m128i_i32[2] = v20;
            v29 = 0;
            v30 = 0;
            sub_1E1A9C0(v24, v23, &v27);
            if ( v26 )
              sub_161E7C0((__int64)&v26, v26);
            result = *(unsigned int *)(a1[45] + 8 * v7);
            goto LABEL_16;
          }
          v11 = *(_QWORD *)(v11 + 32);
        }
        v12 = (unsigned int *)a1[46];
        v13 = (char *)(v8 + 2);
        if ( v12 != v8 + 2 )
        {
          v14 = ((char *)v12 - v13) >> 3;
          if ( (char *)v12 - v13 <= 0 )
          {
            v13 = (char *)a1[46];
          }
          else
          {
            do
            {
              v15 = *(_DWORD *)v13;
              v13 += 8;
              *((_DWORD *)v13 - 4) = v15;
              *((_DWORD *)v13 - 3) = *((_DWORD *)v13 - 1);
              --v14;
            }
            while ( v14 );
            v13 = (char *)a1[46];
          }
        }
        result = (__int64)(v13 - 8);
        --v25;
        a1[46] = result;
        if ( v25 == v5 )
          return result;
      }
    }
  }
  return result;
}
