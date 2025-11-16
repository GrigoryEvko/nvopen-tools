// Function: sub_23B92F0
// Address: 0x23b92f0
//
__int64 __fastcall sub_23B92F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  int v9; // edx
  __int64 *v10; // rax
  __m128i *v11; // rsi
  __m128i *v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  int v20; // edx
  __int64 *v21; // rax
  __m128i *v22; // rsi
  __int64 v23; // rbx
  __m128i *v24; // rdi
  __m128i *v25; // r15
  __m128i *v26; // rax
  unsigned __int64 v27; // rdi
  __int32 v28; // eax
  __m128i *v29; // r15
  __m128i *v30; // rax
  unsigned __int64 v31; // rdi
  __int32 v32; // eax
  __int32 v33; // [rsp+8h] [rbp-58h]
  __int32 v34; // [rsp+8h] [rbp-58h]
  __m128i v35; // [rsp+10h] [rbp-50h] BYREF
  __int64 v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+28h] [rbp-38h]

  result = sub_C96F30();
  if ( result )
  {
    v8 = *(unsigned int *)(a2 + 296);
    v9 = v8;
    if ( *(_DWORD *)(a2 + 300) <= (unsigned int)v8 )
    {
      v25 = (__m128i *)sub_C8D7D0(a2 + 288, a2 + 304, 0, 0x20u, (unsigned __int64 *)&v35, v7);
      v26 = &v25[2 * *(unsigned int *)(a2 + 296)];
      if ( v26 )
      {
        v26->m128i_i64[0] = a1;
        v26[1].m128i_i64[1] = (__int64)off_4CDFC08 + 2;
      }
      sub_BC3E80(a2 + 288, v25);
      v27 = *(_QWORD *)(a2 + 288);
      v28 = v35.m128i_i32[0];
      if ( a2 + 304 != v27 )
      {
        v33 = v35.m128i_i32[0];
        _libc_free(v27);
        v28 = v33;
      }
      ++*(_DWORD *)(a2 + 296);
      *(_QWORD *)(a2 + 288) = v25;
      *(_DWORD *)(a2 + 300) = v28;
    }
    else
    {
      v10 = (__int64 *)(*(_QWORD *)(a2 + 288) + 32 * v8);
      if ( v10 )
      {
        *v10 = a1;
        v10[3] = (__int64)off_4CDFC08 + 2;
        v9 = *(_DWORD *)(a2 + 296);
      }
      *(_DWORD *)(a2 + 296) = v9 + 1;
    }
    v11 = *(__m128i **)(a2 + 432);
    v35.m128i_i64[0] = a1;
    v37 = (__int64)off_4CDFC28 + 2;
    sub_23B8210(a2 + 432, v11, &v35, v5, v6, v7);
    sub_23B7140(v35.m128i_i64);
    v12 = *(__m128i **)(a2 + 576);
    v35.m128i_i64[0] = a1;
    v37 = (__int64)off_4CDFC20 + 2;
    sub_23B86E0(a2 + 576, v12, &v35, v13, v14, v15);
    sub_23B71A0(v35.m128i_i64);
    v19 = *(unsigned int *)(a2 + 728);
    v20 = v19;
    if ( *(_DWORD *)(a2 + 732) <= (unsigned int)v19 )
    {
      v29 = (__m128i *)sub_C8D7D0(a2 + 720, a2 + 736, 0, 0x20u, (unsigned __int64 *)&v35, v18);
      v30 = &v29[2 * *(unsigned int *)(a2 + 728)];
      if ( v30 )
      {
        v30->m128i_i64[0] = a1;
        v30[1].m128i_i64[1] = (__int64)off_4CDFC00 + 2;
      }
      sub_BC3E80(a2 + 720, v29);
      v31 = *(_QWORD *)(a2 + 720);
      v32 = v35.m128i_i32[0];
      if ( a2 + 736 != v31 )
      {
        v34 = v35.m128i_i32[0];
        _libc_free(v31);
        v32 = v34;
      }
      ++*(_DWORD *)(a2 + 728);
      *(_QWORD *)(a2 + 720) = v29;
      *(_DWORD *)(a2 + 732) = v32;
    }
    else
    {
      v21 = (__int64 *)(*(_QWORD *)(a2 + 720) + 32 * v19);
      if ( v21 )
      {
        *v21 = a1;
        v21[3] = (__int64)off_4CDFC00 + 2;
        v20 = *(_DWORD *)(a2 + 728);
      }
      *(_DWORD *)(a2 + 728) = v20 + 1;
    }
    v22 = *(__m128i **)(a2 + 864);
    v35.m128i_i64[0] = a1;
    v37 = (__int64)off_4CDFC18 + 2;
    sub_23B8E70(a2 + 864, v22, &v35, v16, v17, v18);
    result = v37;
    if ( (v37 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v23 = (v37 >> 1) & 1;
      if ( (v37 & 4) != 0 )
      {
        v24 = &v35;
        if ( !(_BYTE)v23 )
          v24 = (__m128i *)v35.m128i_i64[0];
        result = (*(__int64 (__fastcall **)(__m128i *))((v37 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v24);
      }
      if ( !(_BYTE)v23 )
        return sub_C7D6A0(v35.m128i_i64[0], v35.m128i_i64[1], v36);
    }
  }
  return result;
}
