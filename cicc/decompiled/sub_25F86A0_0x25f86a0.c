// Function: sub_25F86A0
// Address: 0x25f86a0
//
__int64 __fastcall sub_25F86A0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 *v2; // rdi
  __int64 v3; // r12
  __int64 result; // rax
  int v5; // edx
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // [rsp+0h] [rbp-E0h]
  _BYTE v9[16]; // [rsp+20h] [rbp-C0h] BYREF
  void (__fastcall *v10)(_QWORD, _QWORD, _QWORD); // [rsp+30h] [rbp-B0h]
  __int64 v11; // [rsp+38h] [rbp-A8h]
  __m128i v12; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE v13[16]; // [rsp+60h] [rbp-80h] BYREF
  void (__fastcall *v14)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-70h]
  __int64 v15; // [rsp+78h] [rbp-68h]
  _BYTE v16[16]; // [rsp+A0h] [rbp-40h] BYREF
  void (__fastcall *v17)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-30h]

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(unsigned __int8 **)(a1 + 16);
  v3 = *(_QWORD *)(v1 + 16);
  if ( (unsigned int)*v2 - 30 > 0xA )
  {
    v7 = sub_B46B10((__int64)v2, 0);
    v6 = v7;
  }
  else
  {
    result = 1;
    if ( !v3 )
      return result;
    sub_AA72C0(&v12, *(_QWORD *)(v3 + 40), 1);
    v10 = 0;
    v8 = _mm_loadu_si128(&v12).m128i_u64[0];
    if ( v14 )
    {
      v14(v9, v13, 2);
      v6 = v8;
      v11 = v15;
      v5 = v8 - 24;
      if ( v8 )
        v6 = v8 - 24;
      v10 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v14;
      if ( v14 )
        v14(v9, v9, 3);
    }
    else
    {
      v6 = v8;
      if ( v8 )
        v6 = v8 - 24;
    }
    if ( v17 )
      v17(v16, v16, 3);
    LODWORD(v7) = (_DWORD)v14;
    if ( v14 )
      v7 = ((__int64 (__fastcall *)(_BYTE *, _BYTE *, __int64))v14)(v13, v13, 3);
  }
  LOBYTE(v7) = v3 != 0;
  LOBYTE(v5) = v6 != v3;
  return v5 & (unsigned int)v7 ^ 1;
}
