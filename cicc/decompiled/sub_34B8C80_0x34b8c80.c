// Function: sub_34B8C80
// Address: 0x34b8c80
//
__int64 __fastcall sub_34B8C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __int64 v7; // r11
  int v9; // r13d
  __int64 result; // rax
  __int64 v13; // rdi
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rdx
  char v18; // al
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  char v23; // dl
  __int64 v24; // r9
  int v25; // r13d
  __int64 i; // r12
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r11
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // rax
  __m128i v38; // xmm0
  __int128 v39; // [rsp-10h] [rbp-B0h]
  __int64 v40; // [rsp-10h] [rbp-B0h]
  char v41; // [rsp+17h] [rbp-89h]
  unsigned __int64 v42; // [rsp+18h] [rbp-88h]
  int v43; // [rsp+20h] [rbp-80h]
  int v45; // [rsp+20h] [rbp-80h]
  __int64 v46; // [rsp+28h] [rbp-78h]
  __int64 v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+30h] [rbp-70h]
  __int64 v49; // [rsp+30h] [rbp-70h]
  char v50; // [rsp+38h] [rbp-68h]
  __int64 v51; // [rsp+38h] [rbp-68h]
  __m128i v52; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v53; // [rsp+50h] [rbp-50h]
  unsigned __int64 v54; // [rsp+58h] [rbp-48h]
  __int64 v55; // [rsp+60h] [rbp-40h]
  __int64 v56; // [rsp+68h] [rbp-38h]

  v7 = a3;
  v9 = a1;
  v48 = a7;
  v50 = BYTE8(a7);
  result = *(unsigned __int8 *)(a3 + 8);
  if ( (_BYTE)result == 15 )
  {
    result = 0;
    if ( a6 )
    {
      v52.m128i_i64[0] = a3;
      result = sub_AE4AC0(a2, a3);
      v7 = v52.m128i_i64[0];
    }
    v13 = *(_QWORD *)(v7 + 16);
    v46 = v13 + 8LL * *(unsigned int *)(v7 + 12);
    if ( v46 != v13 )
    {
      v43 = a4;
      v14 = a6;
      v15 = result;
      v16 = *(_QWORD *)(v7 + 16);
      do
      {
        if ( v15 )
        {
          v17 = *(_QWORD *)(v15 + 16LL * (unsigned int)((v16 - v13) >> 3) + 24);
          v18 = *(_BYTE *)(v15 + 16LL * (unsigned int)((v16 - v13) >> 3) + 32);
          v19 = v48 + v17;
          if ( !v17 )
            v18 = v50;
        }
        else
        {
          v19 = v48;
          v18 = v50;
        }
        v53 = v19;
        LOBYTE(v54) = v18;
        v16 += 8;
        v20 = *(_QWORD *)(v16 - 8);
        v52.m128i_i64[0] = v14;
        result = sub_34B8C80(v9, a2, v20, v43, a5, v14, __PAIR128__(v54, v19));
        v14 = v52.m128i_i64[0];
      }
      while ( v46 != v16 );
    }
  }
  else if ( (_BYTE)result == 16 )
  {
    v47 = *(_QWORD *)(a3 + 24);
    v52.m128i_i8[0] = sub_AE5020(a2, v47);
    v21 = sub_9208B0(a2, v47);
    v56 = v22;
    v55 = v21;
    v41 = v22;
    v42 = ((1LL << v52.m128i_i8[0]) + ((unsigned __int64)(v21 + 7) >> 3) - 1) >> v52.m128i_i8[0] << v52.m128i_i8[0];
    result = *(_QWORD *)(a3 + 32);
    v45 = result;
    if ( (_DWORD)result )
    {
      v23 = v50;
      v24 = a6;
      v25 = 0;
      for ( i = v48; ; i += v42 )
      {
        LOBYTE(v56) = v23;
        *((_QWORD *)&v39 + 1) = v56;
        ++v25;
        v55 = i;
        *(_QWORD *)&v39 = i;
        v52.m128i_i64[0] = v24;
        sub_34B8C80(a1, a2, v47, a4, a5, v24, v39);
        result = v40;
        v24 = v52.m128i_i64[0];
        if ( v45 == v25 )
          break;
        v23 = v41;
        if ( !(i + v42 - v48) )
          v23 = v50;
      }
    }
  }
  else if ( (_BYTE)result != 7 )
  {
    v52.m128i_i64[0] = a3;
    v27 = sub_2D5BAE0(a1, a2, (__int64 *)a3, 0);
    v29 = v52.m128i_i64[0];
    v30 = v27;
    v31 = *(unsigned int *)(a4 + 8);
    v32 = v28;
    if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      v49 = v30;
      v51 = v28;
      sub_C8D5F0(a4, (const void *)(a4 + 16), v31 + 1, 0x10u, v28, v30);
      v31 = *(unsigned int *)(a4 + 8);
      v30 = v49;
      v32 = v51;
      v29 = v52.m128i_i64[0];
    }
    result = *(_QWORD *)a4 + 16 * v31;
    *(_QWORD *)result = v30;
    *(_QWORD *)(result + 8) = v32;
    ++*(_DWORD *)(a4 + 8);
    if ( a5 )
    {
      v33 = sub_336EEB0(a1, a2, v29, 0);
      v34 = *(unsigned int *)(a5 + 8);
      v36 = v35;
      if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        sub_C8D5F0(a5, (const void *)(a5 + 16), v34 + 1, 0x10u, v32, v30);
        v34 = *(unsigned int *)(a5 + 8);
      }
      result = *(_QWORD *)a5 + 16 * v34;
      *(_QWORD *)result = v33;
      *(_QWORD *)(result + 8) = v36;
      ++*(_DWORD *)(a5 + 8);
    }
    if ( a6 )
    {
      v37 = *(unsigned int *)(a6 + 8);
      v38 = _mm_loadu_si128((const __m128i *)&a7);
      if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
      {
        v52 = v38;
        sub_C8D5F0(a6, (const void *)(a6 + 16), v37 + 1, 0x10u, v32, v30);
        v37 = *(unsigned int *)(a6 + 8);
        v38 = _mm_load_si128(&v52);
      }
      result = *(_QWORD *)a6 + 16 * v37;
      *(__m128i *)result = v38;
      ++*(_DWORD *)(a6 + 8);
    }
  }
  return result;
}
