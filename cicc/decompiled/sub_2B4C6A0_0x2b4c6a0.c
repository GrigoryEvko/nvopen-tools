// Function: sub_2B4C6A0
// Address: 0x2b4c6a0
//
__int64 __fastcall sub_2B4C6A0(__int64 a1, const __m128i *a2, __int64 a3, unsigned __int8 *a4)
{
  unsigned int v8; // eax
  unsigned int v9; // r12d
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r10d
  unsigned int i; // eax
  __int64 v15; // rsi
  unsigned int v16; // eax
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  _QWORD *v19; // rdi
  unsigned int v20; // esi
  int v21; // ebx
  __int64 *v22; // rax
  int v23; // edx
  __int64 v24; // rdx
  int v25; // ecx
  unsigned int v26; // esi
  __int64 *v27; // rax
  int v28; // edx
  __int64 v29; // rdx
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 *v31; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v32; // [rsp+28h] [rbp-88h] BYREF
  __int64 v33; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int8 *v34; // [rsp+38h] [rbp-78h]
  __m128i v35[3]; // [rsp+40h] [rbp-70h] BYREF
  char v36; // [rsp+70h] [rbp-40h]

  if ( !a2->m128i_i64[0] )
    return 1;
  LOBYTE(v8) = sub_2B14CA0(a3);
  v9 = v8;
  if ( !(_BYTE)v8 )
    return 1;
  if ( sub_2B14CA0((__int64)a4) )
  {
    v11 = *(unsigned int *)(a1 + 1288);
    v33 = a3;
    v34 = a4;
    v12 = *(_QWORD *)(a1 + 1272);
    if ( (_DWORD)v11 )
    {
      v13 = 1;
      for ( i = (v11 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)
                  | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)))); ; i = (v11 - 1) & v16 )
      {
        v15 = v12 + 24LL * i;
        if ( a3 == *(_QWORD *)v15 && a4 == *(unsigned __int8 **)(v15 + 8) )
          break;
        if ( *(_QWORD *)v15 == -4096 && *(_QWORD *)(v15 + 8) == -4096 )
          goto LABEL_15;
        v16 = v13 + i;
        ++v13;
      }
      if ( v15 != v12 + 24 * v11 )
        return *(unsigned __int8 *)(v15 + 16);
    }
LABEL_15:
    v17 = _mm_loadu_si128(a2 + 1);
    v18 = _mm_loadu_si128(a2 + 2);
    v19 = *(_QWORD **)(a1 + 1296);
    v35[0] = _mm_loadu_si128(a2);
    v35[1] = v17;
    v35[2] = v18;
    v30 = a1 + 1264;
    v36 = 1;
    LOBYTE(v9) = (unsigned __int8)sub_CF63E0(v19, a4, v35, a1 + 1304) != 0;
    if ( (unsigned __int8)sub_2B3DED0(a1 + 1264, &v33, &v32) )
    {
LABEL_16:
      v35[0].m128i_i64[0] = (__int64)a4;
      v35[0].m128i_i64[1] = a3;
      if ( (unsigned __int8)sub_2B3DED0(v30, v35[0].m128i_i64, &v31) )
        return v9;
      v20 = *(_DWORD *)(a1 + 1288);
      v21 = *(_DWORD *)(a1 + 1280);
      v22 = v31;
      ++*(_QWORD *)(a1 + 1264);
      v23 = v21 + 1;
      v32 = v22;
      if ( 4 * (v21 + 1) >= 3 * v20 )
      {
        v20 *= 2;
      }
      else if ( v20 - *(_DWORD *)(a1 + 1284) - v23 > v20 >> 3 )
      {
        goto LABEL_19;
      }
      sub_2B4C3D0(v30, v20);
      sub_2B3DED0(v30, v35[0].m128i_i64, &v32);
      v23 = *(_DWORD *)(a1 + 1280) + 1;
      v22 = v32;
LABEL_19:
      *(_DWORD *)(a1 + 1280) = v23;
      if ( *v22 != -4096 || v22[1] != -4096 )
        --*(_DWORD *)(a1 + 1284);
      *v22 = v35[0].m128i_i64[0];
      v24 = v35[0].m128i_i64[1];
      *((_BYTE *)v22 + 16) = v9;
      v22[1] = v24;
      return v9;
    }
    v25 = *(_DWORD *)(a1 + 1280);
    v26 = *(_DWORD *)(a1 + 1288);
    v27 = v32;
    ++*(_QWORD *)(a1 + 1264);
    v28 = v25 + 1;
    v35[0].m128i_i64[0] = (__int64)v27;
    if ( 4 * (v25 + 1) >= 3 * v26 )
    {
      v26 *= 2;
    }
    else if ( v26 - *(_DWORD *)(a1 + 1284) - v28 > v26 >> 3 )
    {
      goto LABEL_24;
    }
    sub_2B4C3D0(v30, v26);
    sub_2B3DED0(v30, &v33, (__int64 **)v35);
    v28 = *(_DWORD *)(a1 + 1280) + 1;
    v27 = (__int64 *)v35[0].m128i_i64[0];
LABEL_24:
    *(_DWORD *)(a1 + 1280) = v28;
    if ( *v27 != -4096 || v27[1] != -4096 )
      --*(_DWORD *)(a1 + 1284);
    v29 = v33;
    *((_BYTE *)v27 + 16) = v9;
    *v27 = v29;
    v27[1] = (__int64)v34;
    goto LABEL_16;
  }
  return v9;
}
