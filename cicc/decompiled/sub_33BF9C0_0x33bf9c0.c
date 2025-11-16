// Function: sub_33BF9C0
// Address: 0x33bf9c0
//
void __fastcall sub_33BF9C0(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __m128i v22; // xmm0
  char *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  unsigned int v27; // esi
  __int64 *v28; // rdx
  __int64 v29; // r9
  int v30; // edx
  int v31; // r11d
  __int64 v33; // [rsp+8h] [rbp-138h]
  int v34; // [rsp+10h] [rbp-130h]
  int v35; // [rsp+18h] [rbp-128h]
  __m128i v36; // [rsp+20h] [rbp-120h] BYREF
  __m128i v37; // [rsp+30h] [rbp-110h] BYREF
  __int64 v38; // [rsp+40h] [rbp-100h] BYREF
  int v39; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v40[2]; // [rsp+50h] [rbp-F0h] BYREF
  char v41; // [rsp+60h] [rbp-E0h] BYREF
  char *v42; // [rsp+A0h] [rbp-A0h]
  char v43; // [rsp+B8h] [rbp-88h] BYREF
  char *v44; // [rsp+C0h] [rbp-80h]
  char v45; // [rsp+D0h] [rbp-70h] BYREF
  char *v46; // [rsp+E0h] [rbp-60h]
  char v47; // [rsp+F0h] [rbp-50h] BYREF

  v7 = sub_33BF8C0(a1, a2);
  v8 = *(_QWORD *)(a2 + 8);
  v9 = v7;
  v10 = *(_QWORD *)(a1 + 864);
  v36.m128i_i64[0] = v11;
  v34 = v8;
  v12 = *(_QWORD *)(v10 + 16);
  BYTE4(v38) = 0;
  v33 = v12;
  v35 = sub_2E79000(*(__int64 **)(v10 + 40));
  v13 = sub_BD5C60(a2);
  sub_336FEE0((__int64)v40, v13, v33, v35, a3, v34, v38);
  v14 = *(_QWORD *)(a1 + 864);
  v37.m128i_i32[2] = 0;
  v37.m128i_i64[0] = v14 + 288;
  if ( a4 == 215 )
  {
    v24 = *(_QWORD *)(a1 + 960);
    v25 = *(_QWORD *)(v24 + 768);
    v26 = *(unsigned int *)(v24 + 784);
    if ( (_DWORD)v26 )
    {
      v27 = (v26 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = (__int64 *)(v25 + 16LL * (((_DWORD)v26 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
      v29 = *v28;
      if ( a2 == *v28 )
      {
LABEL_21:
        if ( v28 != (__int64 *)(v25 + 16 * v26) )
          a4 = *((_DWORD *)v28 + 2);
      }
      else
      {
        v30 = 1;
        while ( v29 != -4096 )
        {
          v31 = v30 + 1;
          v27 = (v26 - 1) & (v30 + v27);
          v28 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v28;
          if ( a2 == *v28 )
            goto LABEL_21;
          v30 = v31;
        }
      }
    }
  }
  v15 = *(_DWORD *)(a1 + 848);
  v16 = *(_QWORD *)a1;
  v38 = 0;
  v39 = v15;
  if ( v16 )
  {
    if ( &v38 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v38 = v17;
      if ( v17 )
      {
        sub_B96E90((__int64)&v38, v17, 1);
        v14 = *(_QWORD *)(a1 + 864);
      }
    }
  }
  v18 = v36.m128i_i64[0];
  v36.m128i_i64[0] = (__int64)&v38;
  sub_3371E20((__int64)v40, v9, v18, v14, (__int64)&v38, (__int64)&v37, 0, (_BYTE *)a2, a4);
  if ( v38 )
    sub_B91220(v36.m128i_i64[0], v38);
  v21 = *(unsigned int *)(a1 + 424);
  v22 = _mm_load_si128(&v37);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 428) )
  {
    v36 = v22;
    sub_C8D5F0(a1 + 416, (const void *)(a1 + 432), v21 + 1, 0x10u, v19, v20);
    v21 = *(unsigned int *)(a1 + 424);
    v22 = _mm_load_si128(&v36);
  }
  *(__m128i *)(*(_QWORD *)(a1 + 416) + 16 * v21) = v22;
  v23 = v46;
  ++*(_DWORD *)(a1 + 424);
  if ( v23 != &v47 )
    _libc_free((unsigned __int64)v23);
  if ( v44 != &v45 )
    _libc_free((unsigned __int64)v44);
  if ( v42 != &v43 )
    _libc_free((unsigned __int64)v42);
  if ( (char *)v40[0] != &v41 )
    _libc_free(v40[0]);
}
