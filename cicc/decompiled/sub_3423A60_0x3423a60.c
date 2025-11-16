// Function: sub_3423A60
// Address: 0x3423a60
//
void __fastcall sub_3423A60(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, char a6)
{
  __int64 v6; // r14
  __int64 v8; // r15
  __int64 v10; // rbx
  int v11; // ecx
  bool v12; // zf
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rcx
  __int64 *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v22; // [rsp+28h] [rbp-D8h]
  __int64 v23; // [rsp+38h] [rbp-C8h] BYREF
  __m128i v24[2]; // [rsp+40h] [rbp-C0h] BYREF
  _QWORD *v25; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v26; // [rsp+68h] [rbp-98h]
  _BYTE v27[32]; // [rsp+70h] [rbp-90h] BYREF
  void *v28; // [rsp+90h] [rbp-70h] BYREF
  __int64 v29; // [rsp+98h] [rbp-68h]
  __int64 v30; // [rsp+A0h] [rbp-60h]
  __m128i v31; // [rsp+A8h] [rbp-58h] BYREF
  __int64 (__fastcall *v32)(__m128i *, __m128i *, int); // [rsp+B8h] [rbp-48h]
  _QWORD *(__fastcall *v33)(_QWORD **, __int64 *); // [rsp+C0h] [rbp-40h]

  v6 = *(unsigned int *)(a5 + 8);
  v25 = v27;
  v26 = 0x400000000LL;
  if ( (_DWORD)v6 )
  {
    v8 = 0;
    v22 = 8 * v6;
    do
    {
      v10 = *(_QWORD *)(*(_QWORD *)a5 + v8);
      v23 = v10;
      if ( v10 && (v10 != a2 || !a6) )
      {
        v11 = *(_DWORD *)(v10 + 68);
        v24[0].m128i_i64[0] = a5;
        v12 = *(_WORD *)(*(_QWORD *)(v10 + 48) + 16LL * (unsigned int)(v11 - 1)) == 262;
        v13 = *(_QWORD *)(a1 + 64);
        v31 = _mm_loadu_si128(v24);
        v14 = !v12 + v11 - 2;
        v15 = *(_QWORD *)(v13 + 768);
        v30 = v13;
        v29 = v15;
        *(_QWORD *)(v13 + 768) = &v28;
        v28 = &unk_4A366C8;
        v32 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_341E640;
        v33 = sub_341E8E0;
        if ( *(_DWORD *)(v10 + 24) != 2 )
        {
          sub_34161C0(*(_QWORD *)(a1 + 64), v10, v14, a3, a4);
          sub_3421DB0(a3);
        }
        if ( v10 != a2 && !*(_QWORD *)(v10 + 56) )
        {
          v16 = &v25[(unsigned int)v26];
          if ( v16 == sub_341E920(v25, (__int64)v16, &v23) )
          {
            if ( v17 + 1 > (unsigned __int64)HIDWORD(v26) )
            {
              sub_C8D5F0((__int64)&v25, v27, v17 + 1, 8u, v17, v18);
              v16 = &v25[(unsigned int)v26];
            }
            *v16 = v10;
            LODWORD(v26) = v26 + 1;
          }
        }
        v28 = &unk_4A366C8;
        if ( v32 )
          v32(&v31, &v31, 3);
        *(_QWORD *)(v30 + 768) = v29;
      }
      v8 += 8;
    }
    while ( v22 != v8 );
    if ( (_DWORD)v26 )
      sub_33EBD60(*(_QWORD *)(a1 + 64), (__int64)&v25);
    if ( v25 != (_QWORD *)v27 )
      _libc_free((unsigned __int64)v25);
  }
}
