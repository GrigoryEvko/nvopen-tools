// Function: sub_79F800
// Address: 0x79f800
//
__int64 __fastcall sub_79F800(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, unsigned __int64 a5, char *a6)
{
  __int64 v8; // rdi
  unsigned int v11; // r14d
  unsigned __int64 v13; // rdx
  int v15; // [rsp+18h] [rbp-38h] BYREF
  _DWORD v16[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = a2[8].m128i_i64[0];
  v15 = 0;
  v16[0] = 0;
  if ( v8 != a3
    && (!(unsigned int)sub_8D97D0(v8, a3, 0, a4, a5)
     && (sub_7115B0(a2, a3, 0, 1, 1, 1, 0, 1u, 1u, 0, 0, v16, &v15, (_DWORD *)(a4 + 28)), v15)
     || v16[0]) )
  {
    v11 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_687430(0xD2Bu, a4 + 28, a3, a2[8].m128i_i64[0], (_QWORD *)(a1 + 96));
      sub_770D30(a1);
    }
  }
  else
  {
    v13 = a5;
    v11 = 1;
    sub_79CCD0(a1, (__int64)a2, v13, a6, 0);
  }
  return v11;
}
