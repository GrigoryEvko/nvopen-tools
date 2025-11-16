// Function: sub_39BA430
// Address: 0x39ba430
//
__int64 __fastcall sub_39BA430(__int64 a1, _QWORD *a2, unsigned int *a3)
{
  __m128i v5; // xmm3
  __m128i v6; // xmm2
  __m128i v7; // xmm1
  __m128i v8; // xmm0
  unsigned int v9; // r14d
  char v11; // al
  _OWORD *v12; // rdx
  __int64 (*v13)(); // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  _DWORD *v19; // rax
  _DWORD *v20; // rdi
  unsigned int v21; // edx
  int v22; // esi
  unsigned int v23; // ecx
  _OWORD *v24; // [rsp+8h] [rbp-158h]
  __m128i v25[4]; // [rsp+10h] [rbp-150h] BYREF
  __int64 v26; // [rsp+50h] [rbp-110h]
  _OWORD v27[4]; // [rsp+58h] [rbp-108h] BYREF
  __int64 v28; // [rsp+98h] [rbp-C8h]
  __int64 v29; // [rsp+A0h] [rbp-C0h]
  __int64 v30; // [rsp+A8h] [rbp-B8h]
  __int64 v31; // [rsp+B0h] [rbp-B0h]
  __int64 v32; // [rsp+B8h] [rbp-A8h]
  __int64 v33; // [rsp+C0h] [rbp-A0h]
  __int64 v34; // [rsp+C8h] [rbp-98h]
  _BYTE *v35; // [rsp+D0h] [rbp-90h]
  __int64 v36; // [rsp+D8h] [rbp-88h]
  _BYTE v37[128]; // [rsp+E0h] [rbp-80h] BYREF

  v29 = 0;
  v30 = 0;
  v5 = _mm_loadu_si128(xmmword_452E800);
  v6 = _mm_loadu_si128(&xmmword_452E800[1]);
  v31 = 0;
  v7 = _mm_loadu_si128(&xmmword_452E800[2]);
  v8 = _mm_loadu_si128(&xmmword_452E800[3]);
  v32 = 0;
  v35 = v37;
  v33 = 0;
  v26 = unk_452E840;
  v28 = unk_452E840;
  v34 = 0;
  v36 = 0x1000000000LL;
  v25[0] = v5;
  v25[1] = v6;
  v25[2] = v7;
  v25[3] = v8;
  v27[0] = v5;
  v27[1] = v6;
  v27[2] = v7;
  v27[3] = v8;
  sub_1F4B6B0(v25, a2);
  if ( sub_1F4B670((__int64)v25) )
  {
    v9 = sub_1F4BEC0((__int64)v25, a3);
LABEL_3:
    sub_1F4C280(v25, a3);
    sub_39BA2A0(a1, v9, *(double *)v8.m128i_i64);
    goto LABEL_4;
  }
  if ( sub_1F4B690((__int64)v25) )
  {
    v11 = sub_1F4B690((__int64)v25);
    v12 = v27;
    if ( !v11 )
      v12 = 0;
    v13 = *(__int64 (**)())(*a2 + 40LL);
    v24 = v12;
    if ( v13 == sub_1D00B00 )
      BUG();
    v14 = ((__int64 (__fastcall *)(_QWORD *))v13)(a2);
    v15 = *((_QWORD *)v24 + 12);
    v16 = *(unsigned __int16 *)(*(_QWORD *)(v14 + 8) + ((unsigned __int64)*a3 << 6) + 6);
    if ( v15 )
    {
      v17 = *((_QWORD *)v24 + 9);
      v9 = 0;
      v18 = v15 + 10 * v16;
      v19 = (_DWORD *)(v17 + 16LL * *(unsigned __int16 *)(v18 + 2));
      v20 = (_DWORD *)(v17 + 16LL * *(unsigned __int16 *)(v18 + 4));
      if ( v19 != v20 )
      {
        v21 = 0;
        do
        {
          v22 = v19[2];
          v23 = v21 + *v19;
          if ( v9 < v23 )
            v9 = v21 + *v19;
          v21 += v22;
          if ( v22 < 0 )
            v21 = v23;
          v19 += 4;
        }
        while ( v20 != v19 );
      }
    }
    else
    {
      v9 = 1;
    }
    goto LABEL_3;
  }
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
LABEL_4:
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return a1;
}
