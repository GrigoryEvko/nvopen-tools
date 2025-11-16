// Function: sub_1238C00
// Address: 0x1238c00
//
__int64 __fastcall sub_1238C00(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __m128i v7; // xmm0
  _QWORD *v8; // rax
  __int64 v9; // r15
  _QWORD *v10; // r14
  _QWORD *v11; // rax
  unsigned int v12; // [rsp+2Ch] [rbp-74h] BYREF
  __m128i v13; // [rsp+30h] [rbp-70h] BYREF
  _DWORD v14[4]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v15; // [rsp+50h] [rbp-50h] BYREF
  size_t v16; // [rsp+58h] [rbp-48h]
  _QWORD v17[8]; // [rsp+60h] [rbp-40h] BYREF

  v12 = a2;
  v15 = v17;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_120AFE0(a1, 409, "expected 'path' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120B3D0(a1, (__int64)&v15)
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_120AFE0(a1, 410, "expected 'hash' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_120BD00(a1, &v13)
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_120BD00(a1, &v13.m128i_i32[1])
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_120BD00(a1, &v13.m128i_i32[2])
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_120BD00(a1, &v13.m128i_i32[3])
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_120BD00(a1, v14)
    || (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here")
    || (v2 = sub_120AFE0(a1, 13, "expected ')' here"), (_BYTE)v2) )
  {
    v2 = 1;
  }
  else
  {
    v7 = _mm_loadu_si128(&v13);
    v8 = (_QWORD *)sub_9C7D70(*(_QWORD *)(a1 + 352), v15, v16, v4, v5, v6, v7, v7.m128i_i32[0]);
    v9 = *v8;
    v10 = v8;
    v11 = sub_1238B00((_QWORD *)(a1 + 1696), &v12);
    *v11 = v10 + 4;
    v11[1] = v9;
  }
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  return v2;
}
