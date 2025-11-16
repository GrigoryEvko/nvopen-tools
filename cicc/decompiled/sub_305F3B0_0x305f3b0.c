// Function: sub_305F3B0
// Address: 0x305f3b0
//
__int64 __fastcall sub_305F3B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned int v5; // [rsp+Ch] [rbp-14h]

  v2 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), a2, 0);
  BYTE2(v5) = 0;
  return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 32) + 736LL))(
           *(_QWORD *)(a1 + 32),
           *a2,
           v2,
           v3,
           v5);
}
