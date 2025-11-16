// Function: sub_39A9FA0
// Address: 0x39a9fa0
//
__int64 __fastcall sub_39A9FA0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  a1[1] = a2;
  *a1 = &unk_4A3FD60;
  result = *(_QWORD *)(a2 + 272);
  a1[2] = result;
  return result;
}
