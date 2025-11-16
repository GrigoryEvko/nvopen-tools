// Function: sub_E61750
// Address: 0xe61750
//
__int64 __fastcall sub_E61750(_QWORD *a1, __int64 a2, int a3, int a4, int a5, __int64 a6, __int64 a7)
{
  _QWORD *v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 result; // rax

  v10 = (_QWORD *)*a1;
  v11 = v10[36];
  v10[46] += 96LL;
  v12 = (v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10[37] >= v12 + 96 && v11 )
    v10[36] = v12 + 96;
  else
    v12 = sub_9D1E70((__int64)(v10 + 36), 96, 96, 3);
  sub_E81B30(v12, 11, 0);
  *(_DWORD *)(v12 + 32) = a3;
  *(_DWORD *)(v12 + 36) = a4;
  *(_QWORD *)(v12 + 48) = a6;
  *(_DWORD *)(v12 + 40) = a5;
  *(_QWORD *)(v12 + 56) = a7;
  *(_QWORD *)(v12 + 64) = v12 + 88;
  *(_QWORD *)(v12 + 72) = 0;
  *(_QWORD *)(v12 + 80) = 8;
  v13 = *(_QWORD *)(*(_QWORD *)(a2 + 288) + 8LL);
  *(_QWORD *)(v12 + 8) = v13;
  *(_DWORD *)(v12 + 24) = *(_DWORD *)(*(_QWORD *)(a2 + 288) + 24LL) + 1;
  **(_QWORD **)(a2 + 288) = v12;
  *(_QWORD *)(a2 + 288) = v12;
  result = *(_QWORD *)(v13 + 8);
  *(_QWORD *)(result + 8) = v12;
  return result;
}
