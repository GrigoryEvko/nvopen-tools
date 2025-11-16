// Function: sub_16303F0
// Address: 0x16303f0
//
__int64 __fastcall sub_16303F0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 **v10; // rax
  __int64 v11; // rbx
  __int64 result; // rax
  __int64 v14; // rsi
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r13
  unsigned int v19; // esi
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned int v22; // edx
  __int64 v23; // rdi
  unsigned __int8 *v24; // rsi
  unsigned __int8 *v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  int v28; // edi
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // r12
  __int64 v32; // rdi
  int v33; // edx
  int v34; // r9d
  __int64 v35; // rax
  int v36; // r11d
  __int64 v37; // r10
  int v38; // edi
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-38h] BYREF
  __int64 v41[5]; // [rsp+18h] [rbp-28h] BYREF

  v10 = *(__int64 ***)a1;
  v40 = a2;
  v11 = **v10;
  result = *(unsigned int *)(v11 + 424);
  if ( (_DWORD)result )
  {
    v14 = *(_QWORD *)(v11 + 408);
    v15 = (result - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( a1 == *v16 )
    {
LABEL_3:
      result = v14 + 16 * result;
      if ( v16 == (__int64 *)result )
        return result;
      *(_BYTE *)(a1 + 23) &= ~0x10u;
      v18 = v16[1];
      *v16 = -16;
      --*(_DWORD *)(v11 + 416);
      ++*(_DWORD *)(v11 + 420);
      if ( *(_BYTE *)v18 != 2 )
      {
        if ( *(_BYTE *)(v40 + 16) <= 0x10u )
          goto LABEL_6;
LABEL_36:
        sub_16302D0((const __m128i *)(v18 + 8), 0, a3, a4, a5, a6, a7, a8, a9, a10);
        if ( (*(_BYTE *)(v18 + 32) & 1) != 0 )
          return j_j___libc_free_0(v18, 144);
        goto LABEL_13;
      }
      if ( *(_BYTE *)(v40 + 16) <= 0x10u )
      {
        v25 = (unsigned __int8 *)sub_1624210(v40);
        sub_16302D0((const __m128i *)(v18 + 8), v25, a3, a4, a5, a6, v26, v27, a9, a10);
        if ( (*(_BYTE *)(v18 + 32) & 1) == 0 )
LABEL_13:
          j___libc_free_0(*(_QWORD *)(v18 + 40));
        return j_j___libc_free_0(v18, 144);
      }
      if ( *(_BYTE *)(a1 + 16) == 17 )
      {
        v29 = *(_QWORD *)(a1 + 24);
        if ( !v29 )
          goto LABEL_6;
      }
      else
      {
        v35 = *(_QWORD *)(a1 + 40);
        if ( !v35 )
          goto LABEL_6;
        v29 = *(_QWORD *)(v35 + 56);
        if ( !v29 )
          goto LABEL_6;
      }
      if ( !sub_1626D20(v29) )
        goto LABEL_6;
      if ( *(_BYTE *)(v40 + 16) == 17 )
      {
        v30 = *(_QWORD *)(v40 + 24);
        if ( !v30 )
          goto LABEL_6;
      }
      else
      {
        v39 = *(_QWORD *)(v40 + 40);
        if ( !v39 )
          goto LABEL_6;
        v30 = *(_QWORD *)(v39 + 56);
        if ( !v30 )
          goto LABEL_6;
      }
      if ( !sub_1626D20(v30) )
        goto LABEL_6;
      if ( *(_BYTE *)(a1 + 16) == 17 )
      {
        v31 = *(_QWORD *)(a1 + 24);
        if ( !v31 )
          goto LABEL_32;
      }
      else
      {
        v31 = *(_QWORD *)(a1 + 40);
        if ( !v31 || (v31 = *(_QWORD *)(v31 + 56)) == 0 )
        {
LABEL_32:
          if ( *(_BYTE *)(v40 + 16) == 17 )
          {
            v32 = *(_QWORD *)(v40 + 24);
            if ( !v32 )
              goto LABEL_35;
          }
          else
          {
            v32 = *(_QWORD *)(v40 + 40);
            if ( !v32 || (v32 = *(_QWORD *)(v32 + 56)) == 0 )
            {
LABEL_35:
              if ( v32 != v31 )
                goto LABEL_36;
LABEL_6:
              v19 = *(_DWORD *)(v11 + 424);
              if ( v19 )
              {
                v20 = v40;
                v21 = *(_QWORD *)(v11 + 408);
                v22 = (v19 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
                result = v21 + 16LL * v22;
                v23 = *(_QWORD *)result;
                if ( *(_QWORD *)result == v40 )
                {
LABEL_8:
                  v24 = *(unsigned __int8 **)(result + 8);
                  if ( v24 )
                  {
                    sub_16302D0((const __m128i *)(v18 + 8), v24, a3, a4, a5, a6, a7, a8, a9, a10);
                    if ( (*(_BYTE *)(v18 + 32) & 1) == 0 )
                      goto LABEL_13;
                    return j_j___libc_free_0(v18, 144);
                  }
LABEL_22:
                  *(_BYTE *)(v20 + 23) |= 0x10u;
                  *(_QWORD *)(v18 + 136) = v20;
                  *(_QWORD *)(result + 8) = v18;
                  return result;
                }
                v36 = 1;
                v37 = 0;
                while ( v23 != -8 )
                {
                  if ( !v37 && v23 == -16 )
                    v37 = result;
                  v22 = (v19 - 1) & (v36 + v22);
                  result = v21 + 16LL * v22;
                  v23 = *(_QWORD *)result;
                  if ( v40 == *(_QWORD *)result )
                    goto LABEL_8;
                  ++v36;
                }
                v38 = *(_DWORD *)(v11 + 416);
                if ( v37 )
                  result = v37;
                ++*(_QWORD *)(v11 + 400);
                v28 = v38 + 1;
                if ( 4 * v28 < 3 * v19 )
                {
                  if ( v19 - *(_DWORD *)(v11 + 420) - v28 > v19 >> 3 )
                    goto LABEL_19;
                  goto LABEL_18;
                }
              }
              else
              {
                ++*(_QWORD *)(v11 + 400);
              }
              v19 *= 2;
LABEL_18:
              sub_1624050(v11 + 400, v19);
              sub_16215D0(v11 + 400, &v40, v41);
              result = v41[0];
              v20 = v40;
              v28 = *(_DWORD *)(v11 + 416) + 1;
LABEL_19:
              *(_DWORD *)(v11 + 416) = v28;
              if ( *(_QWORD *)result != -8 )
                --*(_DWORD *)(v11 + 420);
              *(_QWORD *)(result + 8) = 0;
              *(_QWORD *)result = v20;
              v20 = v40;
              goto LABEL_22;
            }
          }
          v32 = sub_1626D20(v32);
          goto LABEL_35;
        }
      }
      v31 = sub_1626D20(v31);
      goto LABEL_32;
    }
    v33 = 1;
    while ( v17 != -8 )
    {
      v34 = v33 + 1;
      v15 = (result - 1) & (v33 + v15);
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( a1 == *v16 )
        goto LABEL_3;
      v33 = v34;
    }
  }
  return result;
}
