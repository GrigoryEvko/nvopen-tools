// Function: sub_2F07850
// Address: 0x2f07850
//
__int64 __fastcall sub_2F07850(__int64 *a1, __int64 *a2)
{
  unsigned int v2; // r14d
  __int64 v4; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rsi
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __m128i v9; // xmm0
  __int64 v10; // rax
  __m128i v11; // xmm1
  __int64 v13; // [rsp+0h] [rbp-A0h]
  _QWORD *v14; // [rsp+8h] [rbp-98h] BYREF
  _QWORD v15[2]; // [rsp+18h] [rbp-88h] BYREF
  __m128i v16; // [rsp+28h] [rbp-78h]
  int v17; // [rsp+38h] [rbp-68h]
  __int64 v18; // [rsp+40h] [rbp-60h]
  _QWORD *v19; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v20[2]; // [rsp+58h] [rbp-48h] BYREF
  __m128i v21; // [rsp+68h] [rbp-38h]
  int v22; // [rsp+78h] [rbp-28h]

  v4 = *a2;
  v5 = a2[2];
  v19 = v20;
  v6 = (_BYTE *)a2[1];
  v18 = v4;
  sub_2F07250((__int64 *)&v19, v6, (__int64)&v6[v5]);
  v7 = (_BYTE *)a1[1];
  v8 = a1[2];
  v9 = _mm_loadu_si128((const __m128i *)(a2 + 5));
  v22 = *((_DWORD *)a2 + 14);
  v10 = *a1;
  v14 = v15;
  v13 = v10;
  v21 = v9;
  sub_2F07250((__int64 *)&v14, v7, (__int64)&v7[v8]);
  v11 = _mm_loadu_si128((const __m128i *)(a1 + 5));
  v17 = *((_DWORD *)a1 + 14);
  v16 = v11;
  LOBYTE(v2) = (unsigned int)v13 < (unsigned int)v18;
  if ( (_DWORD)v13 == (_DWORD)v18 )
    LOBYTE(v2) = HIDWORD(v13) < HIDWORD(v18);
  if ( v14 != v15 )
    j_j___libc_free_0((unsigned __int64)v14);
  if ( v19 != v20 )
    j_j___libc_free_0((unsigned __int64)v19);
  return v2;
}
