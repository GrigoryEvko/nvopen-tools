// Function: sub_E8B350
// Address: 0xe8b350
//
__int64 __fastcall sub_E8B350(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 result; // rax

  v7 = *(_QWORD **)(a1 + 8);
  v8 = v7[36];
  v7[46] += 56LL;
  v9 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7[37] >= (unsigned __int64)(v9 + 56) && v8 )
    v7[36] = v9 + 56;
  else
    v9 = sub_9D1E70((__int64)(v7 + 36), 56, 56, 3);
  sub_E81B30(v9, 2, 0);
  *(_BYTE *)(v9 + 30) = 1;
  *(_QWORD *)(v9 + 32) = a3;
  *(_QWORD *)(v9 + 40) = a2;
  *(_QWORD *)(v9 + 48) = a4;
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  *(_QWORD *)(v9 + 8) = v10;
  *(_DWORD *)(v9 + 24) = *(_DWORD *)(*(_QWORD *)(a1 + 288) + 24LL) + 1;
  **(_QWORD **)(a1 + 288) = v9;
  *(_QWORD *)(a1 + 288) = v9;
  result = *(_QWORD *)(v10 + 8);
  *(_QWORD *)(result + 8) = v9;
  return result;
}
