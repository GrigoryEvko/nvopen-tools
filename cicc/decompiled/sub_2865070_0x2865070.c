// Function: sub_2865070
// Address: 0x2865070
//
__int64 __fastcall sub_2865070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // r8
  __int64 v9; // rdi
  unsigned __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v13; // r12
  __int64 *v14; // r14
  __int64 *v15; // rbx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  _BYTE *v19; // rdi
  int v20; // eax
  char v21; // al
  __int64 v22; // rdx
  __int16 v23; // ax
  _BYTE *v24; // rsi
  _BYTE *v25; // rdi
  __int64 v26; // rax
  _QWORD *v27; // rdx
  _BYTE *v28; // rax
  _BYTE *v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rdx
  __m128i v32; // xmm3
  _BYTE *v33; // rdi
  char v34; // al
  __int64 v35; // [rsp+0h] [rbp-E0h]
  __int64 v37; // [rsp+10h] [rbp-D0h]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  char v39; // [rsp+27h] [rbp-B9h]
  __int64 v40; // [rsp+28h] [rbp-B8h]
  __int64 v41; // [rsp+30h] [rbp-B0h]
  __int64 v42; // [rsp+38h] [rbp-A8h]
  char v43; // [rsp+38h] [rbp-A8h]
  _BYTE *v44; // [rsp+38h] [rbp-A8h]
  __int64 v45; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v46; // [rsp+48h] [rbp-98h]
  char v47; // [rsp+58h] [rbp-88h]
  __int64 v48; // [rsp+60h] [rbp-80h]
  _BYTE *v49; // [rsp+68h] [rbp-78h] BYREF
  __int64 v50; // [rsp+70h] [rbp-70h]
  _BYTE v51[32]; // [rsp+78h] [rbp-68h] BYREF
  __int64 v52; // [rsp+98h] [rbp-48h]
  __m128i v53; // [rsp+A0h] [rbp-40h]

  result = *(unsigned int *)(a1 + 1328);
  v7 = *(_QWORD *)(a1 + 1320);
  v8 = (unsigned int)qword_5001308;
  v35 = result;
  v9 = v7 + 2184 * result;
  if ( v7 == v9 )
  {
    if ( (unsigned int)qword_5001308 > 1 )
      return result;
  }
  else
  {
    a6 = (unsigned int)qword_5001308;
    result = v7;
    v10 = 1;
    while ( 1 )
    {
      v11 = *(unsigned int *)(result + 768);
      if ( (unsigned int)v11 >= (unsigned int)qword_5001308 )
        break;
      v10 *= v11;
      if ( (unsigned int)qword_5001308 <= v10 )
        break;
      result += 2184;
      if ( v9 == result )
        return result;
    }
  }
  if ( !v35 )
    return result;
  v37 = 0;
  v38 = 0;
  while ( 2 )
  {
    v12 = v37 + v7;
    v39 = 0;
    v41 = 0;
    v40 = *(unsigned int *)(v12 + 768);
    if ( !*(_DWORD *)(v12 + 768) )
      goto LABEL_44;
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(v12 + 760) + 112 * v41;
        if ( !*(_QWORD *)(v13 + 8) || !*(_BYTE *)(v13 + 16) )
        {
          v14 = *(__int64 **)(v13 + 40);
          v15 = &v14[*(unsigned int *)(v13 + 48)];
          if ( v14 != v15 )
            break;
        }
LABEL_10:
        if ( v40 == ++v41 )
          goto LABEL_42;
      }
      while ( 1 )
      {
        v22 = *v14;
        v23 = *(_WORD *)(*v14 + 24);
        if ( !v23 )
        {
          v24 = v51;
          v25 = v51;
          v45 = *(_QWORD *)v13;
          v46 = _mm_loadu_si128((const __m128i *)(v13 + 8));
          v47 = *(_BYTE *)(v13 + 24);
          v26 = *(_QWORD *)(v13 + 32);
          v49 = v51;
          v48 = v26;
          v50 = 0x400000000LL;
          v20 = *(_DWORD *)(v13 + 48);
          if ( v20 )
          {
            v42 = v22;
            sub_2850210((__int64)&v49, v13 + 40, v22, a4, v8, a6);
            v25 = v49;
            v22 = v42;
            v20 = v50;
            v24 = &v49[8 * (unsigned int)v50];
          }
          v17 = v46.m128i_i64[0];
          v52 = *(_QWORD *)(v13 + 88);
          v53 = _mm_loadu_si128((const __m128i *)(v13 + 96));
          v16 = *(_QWORD *)(v22 + 32);
          a6 = *(unsigned int *)(v16 + 32);
          v27 = *(_QWORD **)(v16 + 24);
          if ( (unsigned int)a6 <= 0x40 )
          {
            if ( (_DWORD)a6 )
            {
              v16 = (unsigned int)(64 - a6);
              v17 = ((__int64)((_QWORD)v27 << (64 - (unsigned __int8)a6)) >> (64 - (unsigned __int8)a6))
                  + v46.m128i_i64[0];
            }
          }
          else
          {
            v17 = *v27 + v46.m128i_i64[0];
          }
          v46.m128i_i64[0] = v17;
          v46.m128i_i8[8] = 0;
          v18 = (__int64)v14 - *(_QWORD *)(v13 + 40);
          v19 = &v25[v18];
          v8 = (__int64)(v19 + 8);
          if ( v19 + 8 != v24 )
          {
            memmove(v19, v19 + 8, (size_t)&v24[-v8]);
            v20 = v50;
          }
          LODWORD(v50) = v20 - 1;
          goto LABEL_20;
        }
        if ( v23 == 15 )
        {
          v28 = *(_BYTE **)(v22 - 8);
          if ( *v28 <= 3u && !*(_QWORD *)v13 )
            break;
        }
LABEL_23:
        if ( v15 == ++v14 )
          goto LABEL_10;
      }
      v45 = 0;
      v29 = v51;
      v46 = _mm_loadu_si128((const __m128i *)(v13 + 8));
      v47 = *(_BYTE *)(v13 + 24);
      v30 = *(_QWORD *)(v13 + 32);
      v50 = 0x400000000LL;
      v16 = (__int64)v51;
      v48 = v30;
      v49 = v51;
      v31 = *(unsigned int *)(v13 + 48);
      if ( (_DWORD)v31 )
      {
        v44 = v28;
        sub_2850210((__int64)&v49, v13 + 40, v31, (__int64)v51, v8, a6);
        v29 = v49;
        v28 = v44;
        LODWORD(v31) = v50;
        v16 = (__int64)&v49[8 * (unsigned int)v50];
      }
      v52 = *(_QWORD *)(v13 + 88);
      v32 = _mm_loadu_si128((const __m128i *)(v13 + 96));
      v45 = (__int64)v28;
      v53 = v32;
      v33 = &v29[(_QWORD)v14 - *(_QWORD *)(v13 + 40)];
      if ( v33 + 8 != (_BYTE *)v16 )
      {
        memmove(v33, v33 + 8, v16 - (_QWORD)(v33 + 8));
        LODWORD(v31) = v50;
      }
      v18 = (unsigned int)(v31 - 1);
      LODWORD(v50) = v18;
LABEL_20:
      v21 = sub_2864E10(v12, (__int64)&v45, v18, v16, v8, a6);
      if ( !v21 )
      {
        if ( v49 != v51 )
          _libc_free((unsigned __int64)v49);
        goto LABEL_23;
      }
      v43 = v21;
      sub_28532A0(v12, (__int64 *)v13);
      --v40;
      v34 = v43;
      if ( v49 != v51 )
      {
        _libc_free((unsigned __int64)v49);
        v34 = v43;
      }
      v39 = v34;
    }
    while ( v40 != v41 );
LABEL_42:
    if ( v39 )
      sub_2855860(v12, v38, a1 + 36280);
LABEL_44:
    result = ++v38;
    v37 += 2184;
    if ( v38 != v35 )
    {
      v7 = *(_QWORD *)(a1 + 1320);
      continue;
    }
    return result;
  }
}
