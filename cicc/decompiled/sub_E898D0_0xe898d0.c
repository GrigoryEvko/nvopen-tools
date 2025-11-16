// Function: sub_E898D0
// Address: 0xe898d0
//
__int64 __fastcall sub_E898D0(__int64 a1)
{
  __int64 result; // rax

  result = sub_E6DCB0(*(_QWORD *)(a1 + 920), "DXBC", 4u, 2);
  *(_QWORD *)(a1 + 24) = result;
  return result;
}
