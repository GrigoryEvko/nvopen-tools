// Function: sub_CAC470
// Address: 0xcac470
//
__int64 __fastcall sub_CAC470(__int64 a1)
{
  __int64 v1; // r13
  __int64 result; // rax
  int v3; // edx
  unsigned int v4; // r12d
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // rdi
  _BYTE *v10; // r13
  unsigned int v11; // eax
  __int64 v12; // rsi
  _BYTE *v13; // r15
  unsigned int v14; // edi
  __int64 v15; // rdx
  unsigned __int64 v16; // r11
  _BYTE *v17; // rax
  size_t v18; // r13
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  _BYTE *v21; // rsi
  _BYTE *v22; // r8
  int v23; // eax
  __int64 v24; // rax
  _BYTE *v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rax
  unsigned __int64 v28; // r15
  int v29; // eax
  __m128i v30; // xmm0
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // rdi
  char v34; // al
  size_t v35; // rbx
  size_t v36; // rdx
  unsigned __int64 v37; // rax
  char v38; // al
  size_t v39; // r13
  __int64 v40; // rdi
  unsigned __int64 v41; // rdx
  int v42; // r11d
  __int64 v43; // [rsp+8h] [rbp-1E8h]
  _BYTE *v44; // [rsp+18h] [rbp-1D8h]
  int v45; // [rsp+18h] [rbp-1D8h]
  unsigned __int8 v46; // [rsp+26h] [rbp-1CAh]
  char v47; // [rsp+27h] [rbp-1C9h]
  unsigned __int8 v48; // [rsp+28h] [rbp-1C8h]
  char v49; // [rsp+35h] [rbp-1BBh] BYREF
  char v50; // [rsp+36h] [rbp-1BAh] BYREF
  unsigned __int8 v51; // [rsp+37h] [rbp-1B9h] BYREF
  unsigned int v52; // [rsp+38h] [rbp-1B8h] BYREF
  _DWORD n[3]; // [rsp+3Ch] [rbp-1B4h] BYREF
  size_t v54; // [rsp+48h] [rbp-1A8h]
  _QWORD src[2]; // [rsp+50h] [rbp-1A0h] BYREF
  int v56; // [rsp+60h] [rbp-190h]
  __m128i v57; // [rsp+68h] [rbp-188h] BYREF
  void *dest; // [rsp+78h] [rbp-178h]
  size_t v59; // [rsp+80h] [rbp-170h]
  _QWORD v60[3]; // [rsp+88h] [rbp-168h] BYREF
  _BYTE *v61; // [rsp+A0h] [rbp-150h] BYREF
  unsigned __int64 v62; // [rsp+A8h] [rbp-148h]
  unsigned __int64 v63; // [rsp+B0h] [rbp-140h]
  _BYTE v64[312]; // [rsp+B8h] [rbp-138h] BYREF

  v1 = a1;
  v51 = 0;
  v46 = sub_CABD20(a1, &v49, &v50, &v52, &v51);
  if ( !v46 )
    return 0;
  result = v51;
  if ( v51 )
    return result;
  v3 = *(_DWORD *)(a1 + 56);
  v4 = 0;
  n[0] = 0;
  v47 = v49;
  if ( v3 >= 0 )
    v4 = *(_DWORD *)(a1 + 56);
  v43 = *(_QWORD *)(a1 + 40);
  if ( v52 )
  {
    v62 = 0;
    v61 = v64;
    v63 = 256;
    goto LABEL_14;
  }
  if ( !(unsigned __int8)sub_CA8580(a1, &v52, v4, n, &v51) )
    return 0;
  v62 = 0;
  v61 = v64;
  v63 = 256;
  if ( !v51 )
  {
LABEL_14:
    while ( 1 )
    {
      v12 = v52;
      result = sub_CA86D0(a1, v52, v4, &v51);
      if ( !(_BYTE)result )
        goto LABEL_47;
      v13 = *(_BYTE **)(a1 + 40);
      v10 = v13;
      if ( v51 )
      {
        v11 = n[0];
        v20 = v62;
        v1 = a1;
        goto LABEL_28;
      }
      sub_CA7D20(a1, (char *)sub_CA6050, 0);
      if ( *(_BYTE **)(a1 + 40) != v13 )
      {
        v14 = n[0];
        v15 = v62;
        if ( n[0] && v47 == 62 )
        {
          v34 = sub_CA8040(a1, v61, v62);
          v14 = n[0];
          if ( v34 )
          {
            v15 = v62;
          }
          else
          {
            if ( n[0] == 1 )
            {
              v38 = sub_CA8040(a1, v13, *(_QWORD *)(a1 + 40) - (_QWORD)v13);
              v39 = n[0];
              v40 = v62;
              v41 = n[0] + v62;
              v42 = v38 == 0 ? 32 : 10;
              if ( v41 > v63 )
              {
                v45 = v38 == 0 ? 32 : 10;
                sub_C8D290((__int64)&v61, v64, v41, 1u, v6, v7);
                v40 = v62;
                v42 = v45;
              }
              if ( v39 )
              {
                memset(&v61[v40], v42, v39);
                v40 = v62;
              }
              v15 = v39 + v40;
              v14 = n[0];
              v62 = v15;
            }
            else
            {
              v15 = v62;
            }
            n[0] = --v14;
          }
        }
        v16 = v14 + v15;
        if ( v16 > v63 )
        {
          sub_C8D290((__int64)&v61, v64, v16, 1u, v6, v7);
          v15 = v62;
        }
        if ( v14 )
        {
          memset(&v61[v15], 10, v14);
          v15 = v62;
        }
        v17 = *(_BYTE **)(a1 + 40);
        v8 = v14 + v15;
        v62 = v8;
        v18 = v17 - v13;
        v19 = v17 - v13 + v8;
        if ( v19 > v63 )
        {
          v44 = v17;
          sub_C8D290((__int64)&v61, v64, v19, 1u, v6, v7);
          v8 = v62;
          v17 = v44;
        }
        if ( v17 != v13 )
        {
          memcpy(&v61[v8], v13, v18);
          v8 = v62;
        }
        n[0] = 0;
        v9 = v18 + v8;
        v10 = *(_BYTE **)(a1 + 40);
        v62 = v9;
      }
      if ( *(_BYTE **)(a1 + 48) == v10 )
      {
        v11 = n[0];
        v20 = v62;
        v1 = a1;
        goto LABEL_67;
      }
      if ( !(unsigned __int8)sub_CA80A0(a1) )
      {
        v11 = n[0];
        v20 = v62;
        v1 = a1;
        v13 = *(_BYTE **)(a1 + 40);
        goto LABEL_28;
      }
      v11 = ++n[0];
      if ( v51 )
      {
        v20 = v62;
        v13 = *(_BYTE **)(a1 + 40);
        v1 = a1;
        goto LABEL_28;
      }
    }
  }
  v11 = n[0];
  v13 = *(_BYTE **)(a1 + 40);
  v20 = 0;
LABEL_28:
  if ( v13 == *(_BYTE **)(v1 + 48) )
  {
LABEL_67:
    if ( !v11 )
    {
      n[0] = 1;
      v11 = 1;
    }
  }
  if ( v50 == 45 )
  {
    if ( v63 >= v20 )
      goto LABEL_60;
    v35 = 0;
    goto LABEL_75;
  }
  if ( v50 != 43 )
  {
    if ( !v20 )
    {
      v21 = v61;
      v22 = v61;
      goto LABEL_33;
    }
    if ( v20 + 1 <= v63 )
    {
      v35 = 1;
      v22 = &v61[v20];
LABEL_59:
      memset(v22, 10, v35);
      v20 = v62 + v35;
LABEL_60:
      v21 = v61;
      v22 = &v61[v20];
      goto LABEL_33;
    }
    ++v20;
    v35 = 1;
    goto LABEL_75;
  }
  v35 = v11;
  v37 = v11 + v20;
  if ( v37 > v63 )
  {
    v20 = v37;
LABEL_75:
    sub_C8D290((__int64)&v61, v64, v20, 1u, v6, v7);
    v20 = v62;
  }
  v21 = v61;
  v22 = &v61[v20];
  if ( v35 )
    goto LABEL_59;
LABEL_33:
  v23 = *(_DWORD *)(v1 + 68);
  v62 = v20;
  if ( !v23 )
    *(_BYTE *)(v1 + 73) = 1;
  v24 = *(_QWORD *)(v1 + 40);
  *(_BYTE *)(v1 + 74) = 0;
  v59 = 0;
  dest = v60;
  v57.m128i_i64[1] = v24 - v43;
  LOBYTE(v60[0]) = 0;
  v56 = 19;
  v57.m128i_i64[0] = v43;
  *(_QWORD *)&n[1] = src;
  sub_CA61F0((__int64 *)&n[1], v21, (__int64)v22);
  v25 = dest;
  if ( *(_QWORD **)&n[1] == src )
  {
    v36 = v54;
    if ( v54 )
    {
      if ( v54 == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, v54);
      v36 = v54;
      v25 = dest;
    }
    v59 = v36;
    v25[v36] = 0;
    v25 = *(_BYTE **)&n[1];
    goto LABEL_39;
  }
  if ( dest == v60 )
  {
    dest = *(void **)&n[1];
    v59 = v54;
    v60[0] = src[0];
    goto LABEL_72;
  }
  v26 = v60[0];
  dest = *(void **)&n[1];
  v59 = v54;
  v60[0] = src[0];
  if ( !v25 )
  {
LABEL_72:
    *(_QWORD *)&n[1] = src;
    v25 = src;
    goto LABEL_39;
  }
  *(_QWORD *)&n[1] = v25;
  src[0] = v26;
LABEL_39:
  v54 = 0;
  *v25 = 0;
  if ( *(_QWORD **)&n[1] != src )
    j_j___libc_free_0(*(_QWORD *)&n[1], src[0] + 1LL);
  v27 = *(_QWORD *)(v1 + 80);
  *(_QWORD *)(v1 + 160) += 72LL;
  v28 = (v27 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(v1 + 88) >= v28 + 72 && v27 )
  {
    *(_QWORD *)(v1 + 80) = v28 + 72;
    if ( !v28 )
    {
      MEMORY[8] = v1 + 176;
      BUG();
    }
  }
  else
  {
    v28 = sub_9D1E70(v1 + 80, 72, 72, 4);
  }
  *(_QWORD *)v28 = 0;
  v29 = v56;
  *(_QWORD *)(v28 + 8) = 0;
  *(_DWORD *)(v28 + 16) = v29;
  v30 = _mm_loadu_si128(&v57);
  *(_QWORD *)(v28 + 40) = v28 + 56;
  *(__m128i *)(v28 + 24) = v30;
  v12 = (__int64)dest;
  sub_CA64F0((__int64 *)(v28 + 40), dest, (__int64)dest + v59);
  v31 = *(_QWORD *)(v1 + 176);
  v32 = *(_QWORD *)v28;
  *(_QWORD *)(v28 + 8) = v1 + 176;
  v31 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v28 = v31 | v32 & 7;
  *(_QWORD *)(v31 + 8) = v28;
  v33 = dest;
  *(_QWORD *)(v1 + 176) = *(_QWORD *)(v1 + 176) & 7LL | v28;
  if ( v33 != v60 )
  {
    v12 = v60[0] + 1LL;
    j_j___libc_free_0(v33, v60[0] + 1LL);
  }
  result = v46;
LABEL_47:
  if ( v61 != v64 )
  {
    v48 = result;
    _libc_free(v61, v12);
    return v48;
  }
  return result;
}
