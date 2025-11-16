// Function: sub_3311FA0
// Address: 0x3311fa0
//
__int64 __fastcall sub_3311FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  const __m128i *v13; // rax
  __int64 v14; // r10
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r10
  __int64 v17; // r9
  __m128i *v18; // rdx
  int v19; // ecx
  _BYTE *v20; // r11
  __int64 v21; // rdi
  __int8 *v23; // [rsp+0h] [rbp-100h]
  const __m128i *v24; // [rsp+8h] [rbp-F8h]
  int v25; // [rsp+10h] [rbp-F0h]
  __int64 (__fastcall **i)(); // [rsp+20h] [rbp-E0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-D8h]
  __int64 v28; // [rsp+30h] [rbp-D0h]
  __int64 v29; // [rsp+38h] [rbp-C8h]
  _BYTE *v30; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+48h] [rbp-B8h]
  _BYTE v32[176]; // [rsp+50h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)a1;
  v29 = a1;
  v7 = *(_QWORD *)(v6 + 768);
  v28 = v6;
  v27 = v7;
  *(_QWORD *)(v6 + 768) = &i;
  v8 = *(_QWORD *)(a2 + 56);
  for ( i = off_4A360B8; v8; v8 = *(_QWORD *)(v8 + 32) )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v8 + 16);
      if ( *(_DWORD *)(v9 + 24) != 328 )
      {
        v30 = *(_BYTE **)(v8 + 16);
        sub_32B3B20(a1 + 568, (__int64 *)&v30);
        if ( *(int *)(v9 + 88) < 0 )
          break;
      }
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_9;
    }
    *(_DWORD *)(v9 + 88) = *(_DWORD *)(a1 + 48);
    v11 = *(unsigned int *)(a1 + 48);
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
    {
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v11 + 1, 8u, a5, v10);
      v11 = *(unsigned int *)(a1 + 48);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v11) = v9;
    ++*(_DWORD *)(a1 + 48);
  }
  do
  {
LABEL_9:
    v12 = *(unsigned int *)(a2 + 64);
    v13 = *(const __m128i **)(a2 + 40);
    v30 = v32;
    v31 = 0x800000000LL;
    v14 = 5 * v12;
    v15 = 5 * v12;
    v16 = 0xCCCCCCCCCCCCCCCDLL * v14;
    v17 = (__int64)&v13->m128i_i64[v15];
    if ( v15 > 40 )
    {
      v23 = &v13->m128i_i8[v15 * 8];
      v24 = v13;
      v25 = v16;
      sub_C8D5F0((__int64)&v30, v32, v16, 0x10u, a5, v17);
      v19 = v31;
      v20 = v30;
      LODWORD(v16) = v25;
      v13 = v24;
      v17 = (__int64)v23;
      v18 = (__m128i *)&v30[16 * (unsigned int)v31];
    }
    else
    {
      v18 = (__m128i *)v32;
      v19 = 0;
      v20 = v32;
    }
    if ( v13 != (const __m128i *)v17 )
    {
      do
      {
        if ( v18 )
          *v18 = _mm_loadu_si128(v13);
        v13 = (const __m128i *)((char *)v13 + 40);
        ++v18;
      }
      while ( (const __m128i *)v17 != v13 );
      v20 = v30;
      v19 = v31;
    }
    v21 = *(_QWORD *)a1;
    LODWORD(v31) = v19 + v16;
    sub_3415F70(v21, a2, v20);
    if ( v30 != v32 )
      _libc_free((unsigned __int64)v30);
  }
  while ( *(_QWORD *)(a2 + 56) );
  sub_32EB240(a1, a2);
  *(_QWORD *)(v28 + 768) = v27;
  return a2;
}
