// Function: sub_C64610
// Address: 0xc64610
//
__int64 *__fastcall sub_C64610(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // r14
  __m128i *v6; // rbx
  __int64 v7; // r13
  __int64 (__fastcall *v8)(__int64, __int64); // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rsi
  int v12; // ecx
  __m128i *v13; // rax
  __int64 v14; // rcx
  _QWORD *v15; // rdi
  __int64 v16; // rax
  char *v18; // rbx
  _QWORD v19[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v21[14]; // [rsp+30h] [rbp-70h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(__int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v5 = *a2;
    *a2 = 0;
    v6 = (__m128i *)v19;
    v7 = *a3;
    v8 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 24LL);
    if ( v8 == sub_9C3610 )
    {
      v21[5] = 0x100000000LL;
      LOBYTE(v20[0]) = 0;
      v21[6] = v19;
      v19[0] = v20;
      v21[0] = &unk_49DD210;
      v19[1] = 0;
      memset(&v21[1], 0, 32);
      sub_CB5980(v21, 0, 0, 0);
      (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v5 + 16LL))(v5, v21);
      v21[0] = &unk_49DD210;
      sub_CB5840(v21);
    }
    else
    {
      v8((__int64)v19, v5);
    }
    v9 = *(unsigned int *)(v7 + 8);
    v10 = *(_QWORD *)v7;
    v11 = v9 + 1;
    v12 = *(_DWORD *)(v7 + 8);
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 12) )
    {
      if ( v10 > (unsigned __int64)v19 || (unsigned __int64)v19 >= v10 + 32 * v9 )
      {
        sub_95D880(v7, v11);
        v9 = *(unsigned int *)(v7 + 8);
        v10 = *(_QWORD *)v7;
        v12 = *(_DWORD *)(v7 + 8);
      }
      else
      {
        v18 = (char *)v19 - v10;
        sub_95D880(v7, v11);
        v10 = *(_QWORD *)v7;
        v9 = *(unsigned int *)(v7 + 8);
        v6 = (__m128i *)&v18[*(_QWORD *)v7];
        v12 = *(_DWORD *)(v7 + 8);
      }
    }
    v13 = (__m128i *)(v10 + 32 * v9);
    if ( v13 )
    {
      v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
      if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
      {
        v13[1] = _mm_loadu_si128(v6 + 1);
      }
      else
      {
        v13->m128i_i64[0] = v6->m128i_i64[0];
        v13[1].m128i_i64[0] = v6[1].m128i_i64[0];
      }
      v14 = v6->m128i_i64[1];
      v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
      v6->m128i_i64[1] = 0;
      v13->m128i_i64[1] = v14;
      v6[1].m128i_i8[0] = 0;
      v12 = *(_DWORD *)(v7 + 8);
    }
    v15 = (_QWORD *)v19[0];
    *(_DWORD *)(v7 + 8) = v12 + 1;
    if ( v15 != v20 )
      j_j___libc_free_0(v15, v20[0] + 1LL);
    *a1 = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  }
  else
  {
    v16 = *a2;
    *a2 = 0;
    *a1 = v16 | 1;
  }
  return a1;
}
