// Function: sub_E7DB20
// Address: 0xe7db20
//
__int64 __fastcall sub_E7DB20(__int64 *a1, __int64 a2, __int64 a3)
{
  const __m128i *v6; // rbx
  __int64 v7; // rdi
  unsigned __int64 v8; // rax
  unsigned int v9; // edx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rax
  int v12; // edi
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // r8
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __m128i *v19; // rax
  __int64 v20; // rax
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // r8
  char *v25; // rbx
  const char *v26; // [rsp+8h] [rbp-90h] BYREF
  char v27; // [rsp+28h] [rbp-70h]
  char v28; // [rsp+29h] [rbp-6Fh]
  __int64 v29; // [rsp+38h] [rbp-60h] BYREF
  int v30; // [rsp+40h] [rbp-58h]
  __int64 v31; // [rsp+48h] [rbp-50h]
  unsigned int v32; // [rsp+50h] [rbp-48h]
  __int16 v33; // [rsp+58h] [rbp-40h]

  v6 = (const __m128i *)&v29;
  v7 = *(_QWORD *)a1[37];
  v26 = ".comment";
  v28 = 1;
  v27 = 3;
  v33 = 257;
  v8 = sub_E71CB0(v7, (size_t *)&v26, 1, 0x30u, 1, (__int64)&v29, 0, -1, 0);
  v9 = *((_DWORD *)a1 + 32);
  v10 = v8;
  v11 = a1[15];
  if ( v9 )
  {
    v14 = v9;
    v13 = 32LL * v9;
    v22 = v11 + v13 - 32;
    v16 = *(_QWORD *)(v22 + 16);
    v9 = *(_DWORD *)(v22 + 24);
    v15 = *(_QWORD *)v22;
    v12 = *(_DWORD *)(v22 + 8);
  }
  else
  {
    v12 = 0;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = 0;
  }
  v32 = v9;
  v17 = v14 + 1;
  v18 = *((unsigned int *)a1 + 33);
  v29 = v15;
  v30 = v12;
  v31 = v16;
  if ( v17 > v18 )
  {
    v23 = (__int64)(a1 + 15);
    v24 = (__int64)(a1 + 17);
    if ( v11 > (unsigned __int64)&v29 || (unsigned __int64)&v29 >= v11 + v13 )
    {
      sub_C8D5F0(v23, a1 + 17, v17, 0x20u, v24, v15);
      v11 = a1[15];
      v13 = 32LL * *((unsigned int *)a1 + 32);
    }
    else
    {
      v25 = (char *)&v29 - v11;
      sub_C8D5F0(v23, a1 + 17, v17, 0x20u, v24, v15);
      v11 = a1[15];
      v6 = (const __m128i *)&v25[v11];
      v13 = 32LL * *((unsigned int *)a1 + 32);
    }
  }
  v19 = (__m128i *)(v13 + v11);
  *v19 = _mm_loadu_si128(v6);
  v19[1] = _mm_loadu_si128(v6 + 1);
  v20 = *a1;
  ++*((_DWORD *)a1 + 32);
  (*(void (__fastcall **)(__int64 *, unsigned __int64, _QWORD))(v20 + 176))(a1, v10, 0);
  if ( !*((_BYTE *)a1 + 6616) )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
    *((_BYTE *)a1 + 6616) = 1;
  }
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 512))(a1, a2, a3);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  return (*(__int64 (__fastcall **)(__int64 *))(*a1 + 168))(a1);
}
