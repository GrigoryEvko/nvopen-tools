// Function: sub_35D8F40
// Address: 0x35d8f40
//
void __fastcall sub_35D8F40(__int64 a1, __m128i a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  const __m128i *v8; // r13
  __m128i *v9; // r12
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // r13
  unsigned __int64 v15; // r14
  __int64 v16; // r12
  void *v17; // rdi
  _QWORD *v18; // r8
  __int64 v19; // rbx
  __int64 v20; // r9
  __int64 *v21; // r10
  unsigned int v22; // edx
  __int64 v23; // r9
  _QWORD *v24; // r8
  const __m128i *v25; // rbx
  const __m128i *v26; // rdi
  int v27; // eax
  bool v28; // si
  __int64 v29; // rbx
  int v30; // eax
  unsigned __int64 v31; // rdx
  __m128i *v32; // rax
  _QWORD *v33; // [rsp+0h] [rbp-70h]
  __int64 *v34; // [rsp+0h] [rbp-70h]
  unsigned int v35; // [rsp+8h] [rbp-68h]
  _QWORD *v36; // [rsp+8h] [rbp-68h]
  _QWORD *v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  _QWORD *v40; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v41; // [rsp+28h] [rbp-48h]
  _QWORD *v42; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v43; // [rsp+38h] [rbp-38h]

  v8 = *(const __m128i **)(a1 + 8);
  v9 = *(__m128i **)a1;
  if ( v8 == *(const __m128i **)a1 )
  {
    v11 = *(_QWORD *)a1;
    v13 = 0;
LABEL_35:
    v15 = 0;
    goto LABEL_28;
  }
  _BitScanReverse64(&v10, 0xCCCCCCCCCCCCCCCDLL * (((char *)v8 - (char *)v9) >> 3));
  sub_35D84F0(*(_QWORD *)a1, *(__m128i **)(a1 + 8), 2LL * (int)(63 - (v10 ^ 0x3F)), a5, a6, a7, a2);
  if ( (char *)v8 - (char *)v9 > 640 )
  {
    v25 = v9 + 40;
    sub_35D7A00(v9, v9 + 40);
    if ( v8 != &v9[40] )
    {
      do
      {
        v26 = v25;
        v25 = (const __m128i *)((char *)v25 + 40);
        sub_35D7970(v26);
      }
      while ( v8 != v25 );
    }
  }
  else
  {
    sub_35D7A00(v9, v8);
  }
  v9 = *(__m128i **)(a1 + 8);
  v11 = *(_QWORD *)a1;
  v12 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v9->m128i_i64 - *(_QWORD *)a1) >> 3);
  v13 = v12;
  if ( !(_DWORD)v12 )
    goto LABEL_35;
  v14 = 0;
  v15 = 0;
  v16 = 40LL * (unsigned int)v12;
  do
  {
    while ( 1 )
    {
      v18 = (_QWORD *)(v11 + v14);
      if ( !(_DWORD)v15 )
        goto LABEL_6;
      v19 = 40LL * (unsigned int)(v15 - 1);
      if ( *(_QWORD *)(v11 + v19 + 24) != v18[3] )
        goto LABEL_6;
      v20 = v18[1];
      v21 = (__int64 *)(*(_QWORD *)(v11 + v19 + 16) + 24LL);
      v41 = *(_DWORD *)(v20 + 32);
      if ( v41 > 0x40 )
      {
        v34 = v21;
        v36 = (_QWORD *)(v11 + v14);
        v38 = v20;
        sub_C43780((__int64)&v40, (const void **)(v20 + 24));
        v21 = v34;
        v18 = v36;
        v20 = v38;
      }
      else
      {
        v40 = *(_QWORD **)(v20 + 24);
      }
      v37 = v18;
      v39 = v20;
      sub_C46B40((__int64)&v40, v21);
      v22 = v41;
      v41 = 0;
      v23 = v39;
      v24 = v37;
      v43 = v22;
      v42 = v40;
      v35 = v22;
      if ( v22 > 0x40 )
        break;
      v11 = *(_QWORD *)a1;
      if ( v40 == (_QWORD *)1 )
        goto LABEL_24;
LABEL_13:
      v18 = (_QWORD *)(v11 + v14);
LABEL_6:
      v14 += 40;
      v17 = (void *)(v11 + 40 * v15);
      v15 = (unsigned int)(v15 + 1);
      memmove(v17, v18, 0x28u);
      v11 = *(_QWORD *)a1;
      if ( v16 == v14 )
        goto LABEL_27;
    }
    v33 = v40;
    v27 = sub_C444A0((__int64)&v42);
    v28 = 0;
    v23 = v39;
    v24 = v37;
    if ( v35 - v27 <= 0x40 )
      v28 = *v33 == 1;
    if ( v42 )
    {
      j_j___libc_free_0_0((unsigned __int64)v42);
      v23 = v39;
      v24 = v37;
      if ( v41 > 0x40 )
      {
        if ( v40 )
        {
          j_j___libc_free_0_0((unsigned __int64)v40);
          v24 = v37;
          v23 = v39;
        }
      }
    }
    v11 = *(_QWORD *)a1;
    if ( !v28 )
      goto LABEL_13;
LABEL_24:
    *(_QWORD *)(v11 + v19 + 16) = v23;
    v29 = *(_QWORD *)a1 + v19;
    v30 = *(_DWORD *)(v29 + 32) + *((_DWORD *)v24 + 8);
    if ( *((unsigned int *)v24 + 8) + (unsigned __int64)*(unsigned int *)(v29 + 32) > 0x80000000 )
      v30 = 0x80000000;
    v14 += 40;
    *(_DWORD *)(v29 + 32) = v30;
    v11 = *(_QWORD *)a1;
  }
  while ( v16 != v14 );
LABEL_27:
  v9 = *(__m128i **)(a1 + 8);
  v31 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v9->m128i_i64 - v11) >> 3);
  v13 = v31;
  if ( v15 > v31 )
  {
    sub_35D8D40((const __m128i **)a1, v15 - v31);
  }
  else
  {
LABEL_28:
    if ( v13 > v15 )
    {
      v32 = (__m128i *)(v11 + 40 * v15);
      if ( v32 != v9 )
        *(_QWORD *)(a1 + 8) = v32;
    }
  }
}
