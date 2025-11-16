// Function: sub_310A840
// Address: 0x310a840
//
__int64 __fastcall sub_310A840(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  result = a1[4];
  a1[3] = a2;
  a1[2] = result;
  return result;
}
