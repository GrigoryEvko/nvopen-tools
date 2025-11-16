// Function: sub_28CECC0
// Address: 0x28cecc0
//
__int64 __fastcall sub_28CECC0(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax

  result = sub_A777F0(0x20u, (__int64 *)(a1 + 72));
  if ( result )
  {
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 8) = 0xFFFFFFFD00000001LL;
    *(_QWORD *)(result + 24) = a2;
    *(_QWORD *)result = &unk_4A219A8;
  }
  *(_DWORD *)(result + 12) = *a2;
  return result;
}
