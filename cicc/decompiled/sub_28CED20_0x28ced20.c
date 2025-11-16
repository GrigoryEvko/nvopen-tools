// Function: sub_28CED20
// Address: 0x28ced20
//
__int64 __fastcall sub_28CED20(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax

  if ( *a2 <= 0x15u )
    return sub_28CECC0(a1, a2);
  result = sub_A777F0(0x20u, (__int64 *)(a1 + 72));
  if ( result )
  {
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 8) = 0xFFFFFFFD00000002LL;
    *(_QWORD *)(result + 24) = a2;
    *(_QWORD *)result = &unk_4A21968;
  }
  *(_DWORD *)(result + 12) = *a2;
  return result;
}
