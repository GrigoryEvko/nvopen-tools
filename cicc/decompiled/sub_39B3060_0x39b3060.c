// Function: sub_39B3060
// Address: 0x39b3060
//
__int64 __fastcall sub_39B3060(
        __int64 a1,
        __int64 a2,
        __int8 *a3,
        size_t a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8,
        _BYTE *a9,
        __int64 a10,
        int a11,
        int a12,
        unsigned int a13)
{
  bool v13; // zf
  __int64 result; // rax

  sub_16FFE70(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  v13 = byte_50575E0 == 0;
  *(_QWORD *)a1 = &unk_4A3FF48;
  *(_DWORD *)(a1 + 592) = a11;
  *(_DWORD *)(a1 + 596) = a12;
  result = a13;
  *(_DWORD *)(a1 + 600) = a13;
  if ( !v13 )
    *(_BYTE *)(a1 + 808) |= 0x10u;
  return result;
}
