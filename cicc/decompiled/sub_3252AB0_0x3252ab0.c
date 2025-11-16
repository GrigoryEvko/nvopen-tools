// Function: sub_3252AB0
// Address: 0x3252ab0
//
__int64 __fastcall sub_3252AB0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  a1[1] = a2;
  *a1 = &unk_4A35E30;
  result = *(_QWORD *)(a2 + 240);
  a1[2] = result;
  return result;
}
