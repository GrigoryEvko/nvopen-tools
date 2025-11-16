// Function: sub_1212A30
// Address: 0x1212a30
//
__int64 __fastcall sub_1212A30(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // r13
  int v7; // eax
  __int64 v8; // rsi
  __int64 v9; // rdi
  int v10; // ecx
  __int64 v11; // rdx
  int *v12; // rax
  int v13; // r8d
  __int64 v14; // rdi
  __int64 v15; // rdi
  unsigned __int64 v16; // rsi
  __int64 v18; // rax
  __int64 v19; // rcx
  __m128i *v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  int v23; // eax
  int v24; // r9d
  _QWORD v25[2]; // [rsp+0h] [rbp-90h] BYREF
  __int64 v26; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v27[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i v28; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v29[4]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v30; // [rsp+60h] [rbp-30h]

  v6 = *a1;
  if ( *(_DWORD *)a2 == 3 )
  {
    v8 = *(_QWORD *)(a2 + 32);
    v14 = sub_BA8B30(*(_QWORD *)(v6 + 344), v8, *(_QWORD *)(a2 + 40));
  }
  else
  {
    v7 = *(_DWORD *)(v6 + 1216);
    v8 = *(unsigned int *)(a2 + 16);
    v9 = *(_QWORD *)(v6 + 1200);
    if ( !v7 )
    {
LABEL_11:
      sub_8FD6D0((__int64)v25, "unknown function '", (_QWORD *)(a2 + 32));
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v25[1]) <= 0x23 )
        sub_4262D8((__int64)"basic_string::append");
      v20 = (__m128i *)sub_2241490(v25, "' referenced by dso_local_equivalent", 36, v19);
      v27[0] = &v28;
      if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
      {
        v28 = _mm_loadu_si128(v20 + 1);
      }
      else
      {
        v27[0] = v20->m128i_i64[0];
        v28.m128i_i64[0] = v20[1].m128i_i64[0];
      }
      v21 = v20->m128i_i64[1];
      v20[1].m128i_i8[0] = 0;
      v27[1] = v21;
      v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
      v20->m128i_i64[1] = 0;
      v22 = *(_QWORD *)(a2 + 8);
      v30 = 260;
      v29[0] = v27;
      sub_11FD800(v6 + 176, v22, (__int64)v29, 1);
      if ( (__m128i *)v27[0] != &v28 )
        j_j___libc_free_0(v27[0], v28.m128i_i64[0] + 1);
      if ( (__int64 *)v25[0] != &v26 )
        j_j___libc_free_0(v25[0], v26 + 1);
      return 1;
    }
    v10 = v7 - 1;
    v11 = (v7 - 1) & (unsigned int)(37 * v8);
    v12 = (int *)(v9 + 16 * v11);
    v13 = *v12;
    if ( (_DWORD)v8 != *v12 )
    {
      v23 = 1;
      while ( v13 != -1 )
      {
        v24 = v23 + 1;
        v11 = v10 & (unsigned int)(v23 + v11);
        v12 = (int *)(v9 + 16LL * (unsigned int)v11);
        v13 = *v12;
        if ( (_DWORD)v8 == *v12 )
          goto LABEL_4;
        v23 = v24;
      }
      goto LABEL_11;
    }
LABEL_4:
    v14 = *((_QWORD *)v12 + 1);
  }
  if ( !v14 )
  {
    v6 = *a1;
    goto LABEL_11;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v14 + 24) + 8LL) != 13 )
  {
    v15 = *a1;
    v16 = *(_QWORD *)(a2 + 8);
    v30 = 259;
    v29[0] = "expected a function, alias to function, or ifunc in dso_local_equivalent";
    sub_11FD800(v15 + 176, v16, (__int64)v29, 1);
    return 1;
  }
  v18 = sub_ACC6E0(v14, v8, v11);
  sub_BD84D0((__int64)a3, v18);
  sub_B30810(a3);
  return 0;
}
