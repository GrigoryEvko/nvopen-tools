// Function: sub_18B8E80
// Address: 0x18b8e80
//
__int64 __fastcall sub_18B8E80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v3; // rax
  __int64 result; // rax

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  v3 = (unsigned __int8 *)sub_1632FA0(*(_QWORD *)(a2 + 40));
  *(_BYTE *)(a1 + 25) = 0;
  result = *v3;
  *(_BYTE *)(a1 + 24) = result;
  return result;
}
