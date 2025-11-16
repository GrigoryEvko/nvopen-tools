// Function: sub_C7D600
// Address: 0xc7d600
//
__int64 __fastcall sub_C7D600(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = 0;
  qword_4F840F0 = v1;
  (*(void (__fastcall **)(_QWORD))(a1 + 8))(*(_QWORD *)a1);
  result = _InterlockedExchange64((volatile __int64 *)a1, 0);
  *(_QWORD *)(a1 + 8) = 0;
  return result;
}
