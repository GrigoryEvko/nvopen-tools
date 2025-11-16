// Function: sub_899FE0
// Address: 0x899fe0
//
__int64 __fastcall sub_899FE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 result; // rax

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 96LL) + 16LL);
  *(_QWORD *)(v2 + 16) = result;
  *(_QWORD *)(result + 8) = v2;
  return result;
}
