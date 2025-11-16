// Function: sub_16BAF60
// Address: 0x16baf60
//
_BYTE *__fastcall sub_16BAF60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  void *v7; // rdi
  unsigned __int64 v8; // r13
  const char *v9; // rsi
  __int64 v10; // rax
  _BYTE *v11; // rbx
  __int64 v12; // r12
  _BYTE *result; // rax
  int v14; // ecx
  __int64 v15; // r8
  unsigned int v16; // esi
  int *v17; // rax
  int v18; // edi
  _BYTE *v19; // rsi
  char v20; // dl
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned int v24; // r15d
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdi
  const char *v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // r15
  __int64 v36; // rax
  int v37; // edx
  int v38; // eax
  int v39; // eax
  int v40; // r9d
  __int64 v41; // rax
  _BYTE *v42; // [rsp+18h] [rbp-F8h]
  unsigned int v43; // [rsp+20h] [rbp-F0h]
  __int64 v44[2]; // [rsp+40h] [rbp-D0h] BYREF
  _QWORD v45[2]; // [rsp+50h] [rbp-C0h] BYREF
  const char *v46; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+68h] [rbp-A8h]
  _QWORD v48[2]; // [rsp+70h] [rbp-A0h] BYREF
  __m128i *v49; // [rsp+80h] [rbp-90h]
  __int64 v50; // [rsp+88h] [rbp-88h]
  __m128i si128; // [rsp+90h] [rbp-80h] BYREF
  __int128 v52; // [rsp+A0h] [rbp-70h]
  __int128 v53; // [rsp+B0h] [rbp-60h]
  __int128 v54; // [rsp+C0h] [rbp-50h] BYREF
  __m128i v55[4]; // [rsp+D0h] [rbp-40h] BYREF

  v43 = a2;
  v4 = sub_16E8C20(a1, a2, a3);
  v5 = *(_QWORD *)(v4 + 24);
  v6 = v4;
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v5) <= 2 )
  {
    v41 = sub_16E7EE0(v4, "  -", 3);
    v7 = *(void **)(v41 + 24);
    v6 = v41;
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 45;
    *(_WORD *)v5 = 8224;
    v7 = (void *)(*(_QWORD *)(v4 + 24) + 3LL);
    *(_QWORD *)(v4 + 24) = v7;
  }
  v8 = *(_QWORD *)(a1 + 32);
  v9 = *(const char **)(a1 + 24);
  if ( v8 > *(_QWORD *)(v6 + 16) - (_QWORD)v7 )
  {
    sub_16E7EE0(v6, v9, *(_QWORD *)(a1 + 32));
    v8 = *(_QWORD *)(a1 + 32);
  }
  else if ( v8 )
  {
    memcpy(v7, v9, *(_QWORD *)(a1 + 32));
    *(_QWORD *)(v6 + 24) += v8;
    v8 = *(_QWORD *)(a1 + 32);
  }
  sub_16B2520(*(_OWORD *)(a1 + 40), v43, v8 + 6);
  v10 = sub_16BAF20();
  v11 = *(_BYTE **)(v10 + 80);
  v12 = v10;
  result = *(_BYTE **)(v10 + 88);
  v42 = result;
  if ( v11 != result )
  {
    while ( 1 )
    {
      v35 = 0x1FFFFFFFE0LL;
      v44[0] = (__int64)v45;
      sub_16BA660(v44, *(_BYTE **)v11, *(_QWORD *)v11 + *((_QWORD *)v11 + 1));
      v36 = sub_C61310(v12 + 32, (__int64)v44);
      v37 = 0;
      if ( v36 != v12 + 40 )
      {
        v37 = *(_DWORD *)(v36 + 64);
        v35 = 32LL * (unsigned int)(v37 - 1);
      }
      v38 = *(_DWORD *)(v12 + 24);
      if ( !v38 )
        goto LABEL_28;
      v14 = v38 - 1;
      v15 = *(_QWORD *)(v12 + 8);
      v16 = (v38 - 1) & (37 * v37);
      v17 = (int *)(v15 + 72LL * v16);
      v18 = *v17;
      if ( v37 != *v17 )
        break;
LABEL_9:
      v19 = (_BYTE *)*((_QWORD *)v17 + 5);
      v52 = *(_OWORD *)(v17 + 2);
      *(_QWORD *)&v53 = *((_QWORD *)v17 + 3);
      v20 = *((_BYTE *)v17 + 32);
      *(_QWORD *)&v54 = v55;
      BYTE8(v53) = v20;
      sub_16BA660((__int64 *)&v54, v19, (__int64)&v19[*((_QWORD *)v17 + 6)]);
LABEL_10:
      v21 = v35 + *(_QWORD *)(v12 + 80);
      v46 = (const char *)v48;
      v22 = *(_QWORD *)v21;
      sub_16BA660((__int64 *)&v46, *(_BYTE **)v21, *(_QWORD *)v21 + *(_QWORD *)(v21 + 8));
      v49 = &si128;
      if ( (__m128i *)v54 == v55 )
      {
        si128 = _mm_load_si128(v55);
      }
      else
      {
        v49 = (__m128i *)v54;
        si128.m128i_i64[0] = v55[0].m128i_i64[0];
      }
      v50 = *((_QWORD *)&v54 + 1);
      v24 = v43 - v47 - 8;
      v25 = sub_16E8C20(&v46, v22, v23);
      v26 = *(_QWORD *)(v25 + 24);
      v27 = v25;
      if ( (unsigned __int64)(*(_QWORD *)(v25 + 16) - v26) <= 4 )
      {
        v27 = sub_16E7EE0(v25, "    =", 5);
      }
      else
      {
        *(_DWORD *)v26 = 538976288;
        *(_BYTE *)(v26 + 4) = 61;
        *(_QWORD *)(v25 + 24) += 5LL;
      }
      v28 = v46;
      sub_16E7EE0(v27, v46, v47);
      v30 = sub_16E8C20(v27, v28, v29);
      v31 = sub_16E8750(v30, v24);
      v32 = *(_QWORD *)(v31 + 24);
      v33 = v31;
      if ( (unsigned __int64)(*(_QWORD *)(v31 + 16) - v32) <= 4 )
      {
        v33 = sub_16E7EE0(v31, " -   ", 5);
      }
      else
      {
        *(_DWORD *)v32 = 538979616;
        *(_BYTE *)(v32 + 4) = 32;
        *(_QWORD *)(v31 + 24) += 5LL;
      }
      v34 = sub_16E7EE0(v33, v49->m128i_i8, v50);
      result = *(_BYTE **)(v34 + 24);
      if ( (unsigned __int64)result >= *(_QWORD *)(v34 + 16) )
      {
        result = (_BYTE *)sub_16E7DE0(v34, 10);
      }
      else
      {
        *(_QWORD *)(v34 + 24) = result + 1;
        *result = 10;
      }
      if ( v49 != &si128 )
        result = (_BYTE *)j_j___libc_free_0(v49, si128.m128i_i64[0] + 1);
      if ( v46 != (const char *)v48 )
        result = (_BYTE *)j_j___libc_free_0(v46, v48[0] + 1LL);
      if ( (_QWORD *)v44[0] != v45 )
        result = (_BYTE *)j_j___libc_free_0(v44[0], v45[0] + 1LL);
      v11 += 32;
      if ( v42 == v11 )
        return result;
    }
    v39 = 1;
    while ( v18 != -1 )
    {
      v40 = v39 + 1;
      v16 = v14 & (v39 + v16);
      v17 = (int *)(v15 + 72LL * v16);
      v18 = *v17;
      if ( v37 == *v17 )
        goto LABEL_9;
      v39 = v40;
    }
LABEL_28:
    v53 = 0xFFFFFFFFFFFFFFFFLL;
    v54 = (unsigned __int64)v55;
    v52 = 0;
    v55[0] = 0;
    goto LABEL_10;
  }
  return result;
}
