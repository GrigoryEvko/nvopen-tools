// Function: sub_1A1C8D0
// Address: 0x1a1c8d0
//
_QWORD *__fastcall sub_1A1C8D0(__int64 *a1, int a2, __int64 a3, __int64 **a4, const __m128i *a5)
{
  _QWORD *v7; // rax
  _BYTE v8[16]; // [rsp+0h] [rbp-30h] BYREF
  __int16 v9; // [rsp+10h] [rbp-20h]

  if ( a4 == *(__int64 ***)a3 )
    return (_QWORD *)a3;
  if ( *(_BYTE *)(a3 + 16) <= 0x10u )
    return (_QWORD *)sub_15A46C0(a2, (__int64 ***)a3, a4, 0);
  v9 = 257;
  v7 = (_QWORD *)sub_15FDBD0(a2, a3, (__int64)a4, (__int64)v8, 0);
  return sub_1A1C7B0(a1, v7, a5);
}
