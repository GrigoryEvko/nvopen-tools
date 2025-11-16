// Function: sub_1212650
// Address: 0x1212650
//
__int64 __fastcall sub_1212650(__int64 a1, _DWORD *a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // r13
  int v7; // eax
  unsigned __int64 v8; // r14
  const char *v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  int v12; // eax
  _QWORD *v13; // rdi
  __int64 v14; // rcx
  __m128i *v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // rsi
  __int64 v18[2]; // [rsp-C8h] [rbp-C8h] BYREF
  _QWORD v19[2]; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD v20[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v21; // [rsp-98h] [rbp-98h] BYREF
  _QWORD v22[2]; // [rsp-88h] [rbp-88h] BYREF
  __m128i v23; // [rsp-78h] [rbp-78h] BYREF
  _QWORD v24[4]; // [rsp-68h] [rbp-68h] BYREF
  __int16 v25; // [rsp-48h] [rbp-48h]

  *a2 = a3;
  result = 0;
  if ( *(_DWORD *)(a1 + 240) == 94 )
  {
    v4 = a1 + 176;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in address space") )
      return 1;
    v7 = *(_DWORD *)(a1 + 240);
    if ( v7 != 512 )
    {
      v8 = *(_QWORD *)(a1 + 232);
      if ( v7 != 529 )
      {
        HIBYTE(v25) = 1;
        v9 = "expected integer or string constant";
LABEL_8:
        v24[0] = v9;
        LOBYTE(v25) = 3;
        sub_11FD800(v4, v8, (__int64)v24, 1);
        return 1;
      }
      if ( (unsigned __int8)sub_120BD00(a1, a2) )
        return 1;
      if ( *a2 > 0xFFFFFFu )
      {
        HIBYTE(v25) = 1;
        v9 = "invalid address space, must be a 24-bit integer";
        goto LABEL_8;
      }
      return sub_120AFE0(a1, 13, "expected ')' in address space");
    }
    v10 = *(_BYTE **)(a1 + 248);
    v11 = *(_QWORD *)(a1 + 256);
    v18[0] = (__int64)v19;
    sub_12060D0(v18, v10, (__int64)&v10[v11]);
    if ( !(unsigned int)sub_2241AC0(v18, "A") )
    {
      *a2 = *(_DWORD *)(*(_QWORD *)(a1 + 344) + 316LL);
      goto LABEL_11;
    }
    if ( !(unsigned int)sub_2241AC0(v18, "G") )
    {
      *a2 = *(_DWORD *)(*(_QWORD *)(a1 + 344) + 324LL);
      goto LABEL_11;
    }
    if ( !(unsigned int)sub_2241AC0(v18, "P") )
    {
      *a2 = *(_DWORD *)(*(_QWORD *)(a1 + 344) + 320LL);
LABEL_11:
      v12 = sub_1205200(v4);
      v13 = (_QWORD *)v18[0];
      *(_DWORD *)(a1 + 240) = v12;
      if ( v13 != v19 )
        j_j___libc_free_0(v13, v19[0] + 1LL);
      return sub_120AFE0(a1, 13, "expected ')' in address space");
    }
    sub_8FD6D0((__int64)v20, "invalid symbolic addrspace '", v18);
    if ( v20[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v15 = (__m128i *)sub_2241490(v20, "'", 1, v14);
    v22[0] = &v23;
    if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
    {
      v23 = _mm_loadu_si128(v15 + 1);
    }
    else
    {
      v22[0] = v15->m128i_i64[0];
      v23.m128i_i64[0] = v15[1].m128i_i64[0];
    }
    v16 = v15->m128i_i64[1];
    v15[1].m128i_i8[0] = 0;
    v22[1] = v16;
    v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
    v15->m128i_i64[1] = 0;
    v17 = *(_QWORD *)(a1 + 232);
    v25 = 260;
    v24[0] = v22;
    sub_11FD800(v4, v17, (__int64)v24, 1);
    if ( (__m128i *)v22[0] != &v23 )
      j_j___libc_free_0(v22[0], v23.m128i_i64[0] + 1);
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0], v21 + 1);
    result = 1;
    if ( (_QWORD *)v18[0] != v19 )
    {
      j_j___libc_free_0(v18[0], v19[0] + 1LL);
      return 1;
    }
  }
  return result;
}
