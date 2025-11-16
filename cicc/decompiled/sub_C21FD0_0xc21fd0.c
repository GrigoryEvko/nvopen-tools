// Function: sub_C21FD0
// Address: 0xc21fd0
//
__int64 __fastcall sub_C21FD0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  int v4; // eax
  __int64 (__fastcall ***v5)(); // rdx
  __int64 v7; // rax
  const __m128i *v8; // rax
  __m128i v9; // xmm0
  _QWORD v10[2]; // [rsp+0h] [rbp-40h] BYREF
  char v11; // [rsp+10h] [rbp-30h]

  sub_C21E40((__int64)v10, a2);
  if ( (v11 & 1) != 0 )
  {
    v4 = v10[0];
    v5 = (__int64 (__fastcall ***)())v10[1];
    if ( LODWORD(v10[0]) )
      goto LABEL_3;
  }
  v7 = v10[0];
  if ( v10[0] >= (unsigned __int64)((__int64)(a2[29] - a2[28]) >> 4) )
  {
    v5 = sub_C1AFD0();
    v4 = 8;
LABEL_3:
    *(_DWORD *)a1 = v4;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v5;
    return a1;
  }
  if ( a3 )
    *a3 = v10[0];
  v8 = (const __m128i *)(a2[28] + 16 * v7);
  *(_BYTE *)(a1 + 16) &= ~1u;
  v9 = _mm_loadu_si128(v8);
  *(__m128i *)a1 = v9;
  return a1;
}
