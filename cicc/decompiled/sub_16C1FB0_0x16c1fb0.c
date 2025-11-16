// Function: sub_16C1FB0
// Address: 0x16c1fb0
//
__int64 __fastcall sub_16C1FB0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = 0;
  qword_4FA04E0 = v1;
  (*(void (__fastcall **)(_QWORD))(a1 + 8))(*(_QWORD *)a1);
  result = _InterlockedExchange64((volatile __int64 *)a1, 0);
  *(_QWORD *)(a1 + 8) = 0;
  return result;
}
