// Function: sub_12BF030
// Address: 0x12bf030
//
__int64 *__fastcall sub_12BF030(__int64 *a1, __int64 **a2, __int64 *a3)
{
  __int64 *v5; // r14
  __int64 v6; // rbx
  __int64 *(__fastcall *v7)(__int64 *, __int64 *); // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __m128i *v12; // rax
  __int64 *v13; // rax
  __m128i *v15; // rdi
  __m128i *v16; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v17; // [rsp+18h] [rbp-98h]
  __m128i v18; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v19[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v20[2]; // [rsp+40h] [rbp-70h] BYREF
  void *v21; // [rsp+50h] [rbp-60h] BYREF
  __int64 v22; // [rsp+58h] [rbp-58h]
  __int64 v23; // [rsp+60h] [rbp-50h]
  __int64 v24; // [rsp+68h] [rbp-48h]
  int v25; // [rsp+70h] [rbp-40h]
  _QWORD *v26; // [rsp+78h] [rbp-38h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, void *))(**a2 + 48))(*a2, &unk_4FA032B) )
  {
    v5 = *a2;
    *a2 = 0;
    v6 = *a3;
    v7 = *(__int64 *(__fastcall **)(__int64 *, __int64 *))(*v5 + 24);
    if ( v7 == sub_12BD5E0 )
    {
      LOBYTE(v20[0]) = 0;
      v19[0] = v20;
      v19[1] = 0;
      v25 = 1;
      v24 = 0;
      v23 = 0;
      v22 = 0;
      v21 = &unk_49EFBE0;
      v26 = v19;
      (*(void (__fastcall **)(__int64 *, void **))(*v5 + 16))(v5, &v21);
      if ( v24 != v22 )
        sub_16E7BA0(&v21);
      v16 = &v18;
      sub_12BCB70((__int64 *)&v16, (_BYTE *)*v26, *v26 + v26[1]);
      sub_16E7BC0(&v21);
      if ( (_QWORD *)v19[0] != v20 )
        j_j___libc_free_0(v19[0], v20[0] + 1LL);
    }
    else
    {
      v7((__int64 *)&v16, v5);
    }
    v11 = *(unsigned int *)(v6 + 8);
    if ( (unsigned int)v11 >= *(_DWORD *)(v6 + 12) )
    {
      sub_12BE710(v6, 0, v11, v8, v9, v10);
      LODWORD(v11) = *(_DWORD *)(v6 + 8);
    }
    v12 = (__m128i *)(*(_QWORD *)v6 + 32LL * (unsigned int)v11);
    if ( v12 )
    {
      v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
      if ( v16 == &v18 )
      {
        v12[1] = _mm_load_si128(&v18);
      }
      else
      {
        v12->m128i_i64[0] = (__int64)v16;
        v12[1].m128i_i64[0] = v18.m128i_i64[0];
      }
      v12->m128i_i64[1] = v17;
      v17 = 0;
      v18.m128i_i8[0] = 0;
      ++*(_DWORD *)(v6 + 8);
    }
    else
    {
      v15 = v16;
      *(_DWORD *)(v6 + 8) = v11 + 1;
      if ( v15 != &v18 )
        j_j___libc_free_0(v15, v18.m128i_i64[0] + 1);
    }
    *a1 = 1;
    (*(void (__fastcall **)(__int64 *))(*v5 + 8))(v5);
  }
  else
  {
    v13 = *a2;
    *a2 = 0;
    *a1 = (unsigned __int64)v13 | 1;
  }
  return a1;
}
