// Function: sub_E8B260
// Address: 0xe8b260
//
__int64 __fastcall sub_E8B260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 result; // rax

  v8 = *(_QWORD **)(a1 + 8);
  v9 = v8[36];
  v8[46] += 64LL;
  v10 = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8[37] >= (unsigned __int64)(v10 + 64) && v9 )
    v8[36] = v10 + 64;
  else
    v10 = sub_9D1E70((__int64)(v8 + 36), 64, 64, 3);
  sub_E81B30(v10, 3, 0);
  *(_QWORD *)(v10 + 32) = a2;
  *(_QWORD *)(v10 + 40) = a3;
  *(_QWORD *)(v10 + 48) = a4;
  *(_QWORD *)(v10 + 56) = a5;
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  *(_QWORD *)(v10 + 8) = v11;
  *(_DWORD *)(v10 + 24) = *(_DWORD *)(*(_QWORD *)(a1 + 288) + 24LL) + 1;
  **(_QWORD **)(a1 + 288) = v10;
  *(_QWORD *)(a1 + 288) = v10;
  result = *(_QWORD *)(v11 + 8);
  *(_QWORD *)(result + 8) = v10;
  return result;
}
