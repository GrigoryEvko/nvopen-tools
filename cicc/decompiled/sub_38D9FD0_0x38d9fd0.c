// Function: sub_38D9FD0
// Address: 0x38d9fd0
//
unsigned __int64 __fastcall sub_38D9FD0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        int a6,
        int a7,
        int a8,
        __int64 a9)
{
  unsigned __int64 result; // rax
  char v13; // dl
  char v14; // dl

  sub_38D76F0(a1, 2, a8, a9);
  *(_DWORD *)(a1 + 184) = a6;
  *(_QWORD *)a1 = &unk_4A3E5C0;
  *(_DWORD *)(a1 + 188) = a7;
  for ( result = 0; result != 16; ++result )
  {
    v13 = 0;
    if ( result < a3 )
      v13 = *(_BYTE *)(a2 + result);
    *(_BYTE *)(a1 + result + 152) = v13;
    v14 = 0;
    if ( result < a5 )
      v14 = *(_BYTE *)(a4 + result);
    *(_BYTE *)(a1 + result + 168) = v14;
  }
  return result;
}
