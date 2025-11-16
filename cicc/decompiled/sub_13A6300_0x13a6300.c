// Function: sub_13A6300
// Address: 0x13a6300
//
__int64 __fastcall sub_13A6300(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 result; // rax

  v5 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)a1 = 2;
  v6 = sub_1456040(a2);
  v7 = sub_145CF80(v5, v6, 1, 0);
  v8 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = v7;
  v9 = sub_1480620(v8, v7, 0);
  v10 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 24) = v9;
  result = sub_1480620(v10, a2, 0);
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 32) = result;
  return result;
}
