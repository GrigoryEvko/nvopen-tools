// Function: sub_1AF45F0
// Address: 0x1af45f0
//
__int64 __fastcall sub_1AF45F0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v15; // r8
  __int64 v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // r8
  _QWORD *v21; // r9
  double v22; // xmm4_8
  double v23; // xmm5_8
  unsigned __int8 v24; // bl
  __int64 v26; // r15
  int v27; // r14d
  __int64 *v28; // rax
  __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rcx
  _QWORD *v33; // r8
  _QWORD *v34; // r9
  __int64 v35; // rdi
  __int64 v36; // [rsp+0h] [rbp-70h]
  unsigned __int8 v37; // [rsp+Fh] [rbp-61h]
  __m128i v38; // [rsp+10h] [rbp-60h] BYREF
  __int64 v39; // [rsp+20h] [rbp-50h]
  __int64 v40; // [rsp+28h] [rbp-48h]
  __int64 v41; // [rsp+30h] [rbp-40h]

  v37 = sub_1AE9990(a1, a4);
  if ( v37 )
  {
    v26 = 0;
    sub_1AEAA40(a1);
    v27 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    if ( v27 )
    {
      do
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v28 = (__int64 *)(*(_QWORD *)(a1 - 8) + 24 * v26);
        else
          v28 = (__int64 *)(a1 + 24 * (v26 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
        v35 = *v28;
        if ( *v28 )
        {
          v29 = v28[1];
          v30 = v28[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v30 = v29;
          if ( v29 )
            *(_QWORD *)(v29 + 16) = *(_QWORD *)(v29 + 16) & 3LL | v30;
        }
        *v28 = 0;
        if ( !*(_QWORD *)(v35 + 8) && a1 != v35 && *(_BYTE *)(v35 + 16) > 0x17u )
        {
          v38.m128i_i64[0] = v35;
          if ( sub_1AE9990(v35, a4) )
            sub_1AF42F0(a2, &v38, v31, v32, v33, v34);
        }
        ++v26;
      }
      while ( v27 != (_DWORD)v26 );
    }
    sub_15F20C0((_QWORD *)a1);
  }
  else
  {
    v38 = (__m128i)a3;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v36 = sub_13E3350(a1, &v38, 0, 1, v15);
    if ( v36 )
    {
      v16 = *(_QWORD *)(a1 + 8);
      if ( v16 )
      {
        do
        {
          v17 = sub_1648700(v16);
          if ( (_QWORD *)a1 != v17 )
          {
            v38.m128i_i64[0] = (__int64)v17;
            sub_1AF42F0(a2, &v38, v18, v19, v20, v21);
          }
          v16 = *(_QWORD *)(v16 + 8);
        }
        while ( v16 );
        if ( *(_QWORD *)(a1 + 8) )
        {
          sub_164D160(a1, v36, a5, a6, a7, a8, v22, v23, a11, a12);
          v37 = 1;
        }
      }
      v24 = sub_1AE9990(a1, a4);
      if ( v24 )
      {
        sub_15F20C0((_QWORD *)a1);
        return v24;
      }
    }
  }
  return v37;
}
