// Function: sub_267B490
// Address: 0x267b490
//
__int64 *__fastcall sub_267B490(__int64 a1)
{
  __int64 v1; // rdx
  __int64 *result; // rax
  _OWORD *v4; // rbx
  __int64 v5; // r14
  __int64 i; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int64 *v17; // r15
  __int64 v18; // r8
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // r15
  __int64 v22; // r8
  unsigned __int64 *v23; // r12
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // [rsp+10h] [rbp-430h]
  __int64 *v28; // [rsp+20h] [rbp-420h]
  __int64 *v29; // [rsp+30h] [rbp-410h]
  _OWORD *v30; // [rsp+48h] [rbp-3F8h] BYREF
  _QWORD v31[2]; // [rsp+50h] [rbp-3F0h] BYREF
  _OWORD v32[4]; // [rsp+60h] [rbp-3E0h] BYREF
  __int64 v33; // [rsp+A0h] [rbp-3A0h]
  _QWORD v34[10]; // [rsp+B0h] [rbp-390h] BYREF
  unsigned __int64 *v35; // [rsp+100h] [rbp-340h]
  unsigned int v36; // [rsp+108h] [rbp-338h]
  char v37; // [rsp+110h] [rbp-330h] BYREF
  _QWORD v38[10]; // [rsp+260h] [rbp-1E0h] BYREF
  unsigned __int64 *v39; // [rsp+2B0h] [rbp-190h]
  unsigned int v40; // [rsp+2B8h] [rbp-188h]
  char v41; // [rsp+2C0h] [rbp-180h] BYREF

  v1 = *(_QWORD *)(a1 + 40);
  v31[0] = 0x100000000LL;
  v31[1] = 0x300000002LL;
  result = *(__int64 **)v1;
  v28 = *(__int64 **)v1;
  v27 = *(_QWORD *)v1 + 8LL * *(unsigned int *)(v1 + 8);
  if ( v27 != *(_QWORD *)v1 )
  {
    do
    {
      v4 = v31;
      v5 = *v28;
      for ( i = 0; ; i = *(int *)v4 )
      {
        v7 = *(_QWORD *)(a1 + 64);
        v8 = *(_QWORD *)(a1 + 72) + 72 * i;
        v9 = _mm_loadu_si128((const __m128i *)(v8 + 34584));
        v10 = _mm_loadu_si128((const __m128i *)(v8 + 34600));
        v11 = _mm_loadu_si128((const __m128i *)(v8 + 34616));
        v12 = _mm_loadu_si128((const __m128i *)(v8 + 34632));
        v13 = *(_QWORD *)(v8 + 34648);
        v32[0] = v9;
        v32[1] = v10;
        v33 = v13;
        v32[2] = v11;
        v32[3] = v12;
        v30 = v32;
        v14 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(a1 + 56))(v7, v5);
        v15 = *v14;
        v29 = v14;
        v16 = sub_B2BE50(*v14);
        if ( sub_B6EA50(v16)
          || (v25 = sub_B2BE50(v15),
              v26 = sub_B6F970(v25),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v26 + 48LL))(v26)) )
        {
          sub_B179F0((__int64)v38, (__int64)"openmp-opt", (__int64)"OpenMPICVTracker", 16, v5);
          sub_267B140((__int64)v34, (__int64)&v30, (__int64)v38);
          v17 = v39;
          v18 = 10LL * v40;
          v38[0] = &unk_49D9D40;
          v19 = &v39[v18];
          if ( v39 != &v39[v18] )
          {
            do
            {
              v19 -= 10;
              v20 = v19[4];
              if ( (unsigned __int64 *)v20 != v19 + 6 )
                j_j___libc_free_0(v20);
              if ( (unsigned __int64 *)*v19 != v19 + 2 )
                j_j___libc_free_0(*v19);
            }
            while ( v17 != v19 );
            v19 = v39;
          }
          if ( v19 != (unsigned __int64 *)&v41 )
            _libc_free((unsigned __int64)v19);
          sub_1049740(v29, (__int64)v34);
          v21 = v35;
          v34[0] = &unk_49D9D40;
          v22 = 10LL * v36;
          v23 = &v35[v22];
          if ( v35 != &v35[v22] )
          {
            do
            {
              v23 -= 10;
              v24 = v23[4];
              if ( (unsigned __int64 *)v24 != v23 + 6 )
                j_j___libc_free_0(v24);
              if ( (unsigned __int64 *)*v23 != v23 + 2 )
                j_j___libc_free_0(*v23);
            }
            while ( v21 != v23 );
            v23 = v35;
          }
          if ( v23 != (unsigned __int64 *)&v37 )
            _libc_free((unsigned __int64)v23);
        }
        v4 = (_OWORD *)((char *)v4 + 4);
        if ( v4 == v32 )
          break;
      }
      result = ++v28;
    }
    while ( (__int64 *)v27 != v28 );
  }
  return result;
}
