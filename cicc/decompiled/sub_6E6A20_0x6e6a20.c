// Function: sub_6E6A20
// Address: 0x6e6a20
//
__int64 __fastcall sub_6E6A20(__int64 a1)
{
  __int64 result; // rax

  result = (unsigned int)sub_8D2310(*(_QWORD *)a1) == 0 ? 1 : 3;
  *(_BYTE *)(a1 + 17) = result;
  return result;
}
