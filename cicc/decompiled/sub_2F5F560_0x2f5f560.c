// Function: sub_2F5F560
// Address: 0x2f5f560
//
__int64 __fastcall sub_2F5F560(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rax
  __int64 v7; // rax
  __int64 result; // rax

  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)a1 = &unk_4A2B2B0;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 32);
  v6 = *(_QWORD **)(a3 + 24);
  *(_QWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 32) = *v6;
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  *(_QWORD *)(a1 + 56) = a4;
  *(_QWORD *)(a1 + 40) = v7;
  *(_QWORD *)(a1 + 48) = a3 + 48;
  result = *(unsigned __int16 *)(a3 + 29080);
  *(_WORD *)(a1 + 64) = result;
  return result;
}
