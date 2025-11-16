// Function: sub_1D49D40
// Address: 0x1d49d40
//
void __fastcall sub_1D49D40(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, char a6)
{
  __int64 v6; // r13
  __int64 v9; // r15
  __int64 v11; // rbx
  int v12; // ecx
  __m128i v13; // xmm0
  bool v14; // zf
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rcx
  __int64 *v18; // rdx
  __int64 *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v25; // [rsp+28h] [rbp-C8h]
  __m128i v26[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 *v27; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+58h] [rbp-98h]
  _BYTE v29[32]; // [rsp+60h] [rbp-90h] BYREF
  void *v30; // [rsp+80h] [rbp-70h] BYREF
  __int64 v31; // [rsp+88h] [rbp-68h]
  __int64 v32; // [rsp+90h] [rbp-60h]
  __m128i v33; // [rsp+98h] [rbp-58h] BYREF
  __int64 (__fastcall *v34)(__m128i *, __m128i *, int); // [rsp+A8h] [rbp-48h]
  _QWORD *(__fastcall *v35)(_QWORD **, __int64 *); // [rsp+B0h] [rbp-40h]

  v6 = *(unsigned int *)(a5 + 8);
  v27 = (__int64 *)v29;
  v28 = 0x400000000LL;
  if ( (_DWORD)v6 )
  {
    v25 = 8 * v6;
    v9 = 0;
    do
    {
      v11 = *(_QWORD *)(*(_QWORD *)a5 + v9);
      if ( v11 && (v11 != a2 || !a6) )
      {
        v12 = *(_DWORD *)(v11 + 60);
        v26[0].m128i_i64[0] = a5;
        v13 = _mm_loadu_si128(v26);
        v14 = *(_BYTE *)(*(_QWORD *)(v11 + 40) + 16LL * (unsigned int)(v12 - 1)) == 111;
        v15 = *(_QWORD *)(a1 + 272);
        v34 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_1D463A0;
        v35 = sub_1D464F0;
        v16 = !v14 + v12 - 2;
        v17 = *(_QWORD *)(v15 + 664);
        v32 = v15;
        v33 = v13;
        v31 = v17;
        *(_QWORD *)(v15 + 664) = &v30;
        v14 = *(_WORD *)(v11 + 24) == 2;
        v30 = &unk_49F9A08;
        if ( !v14 )
        {
          sub_1D44C70(*(_QWORD *)(a1 + 272), v11, v16, a3, a4);
          sub_1D49010(a3);
        }
        if ( v11 != a2 && !*(_QWORD *)(v11 + 48) )
        {
          v18 = v27;
          v19 = &v27[(unsigned int)v28];
          if ( v27 == v19 )
            goto LABEL_23;
          v20 = 0;
          do
          {
            v21 = v11 == *v18++;
            v20 += v21;
          }
          while ( v19 != v18 );
          if ( !v20 )
          {
LABEL_23:
            if ( (unsigned int)v28 >= HIDWORD(v28) )
            {
              sub_16CD150((__int64)&v27, v29, 0, 8, v28, a6);
              v19 = &v27[(unsigned int)v28];
            }
            *v19 = v11;
            LODWORD(v28) = v28 + 1;
          }
        }
        v30 = &unk_49F9A08;
        if ( v34 )
          v34(&v33, &v33, 3);
        *(_QWORD *)(v32 + 664) = v31;
      }
      v9 += 8;
    }
    while ( v25 != v9 );
    if ( (_DWORD)v28 )
      sub_1D2D860(*(_QWORD *)(a1 + 272), (__int64)&v27);
    if ( v27 != (__int64 *)v29 )
      _libc_free((unsigned __int64)v27);
  }
}
