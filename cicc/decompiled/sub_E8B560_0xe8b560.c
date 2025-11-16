// Function: sub_E8B560
// Address: 0xe8b560
//
__int64 __fastcall sub_E8B560(__int64 a1, unsigned __int8 a2, __int64 a3, int a4, int a5)
{
  int v7; // r12d
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 result; // rax

  v7 = a5;
  v8 = *(_QWORD **)(a1 + 8);
  if ( !a5 )
    v7 = 1LL << a2;
  v9 = v8[36];
  v8[46] += 56LL;
  v10 = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8[37] >= (unsigned __int64)(v10 + 56) && v9 )
    v8[36] = v10 + 56;
  else
    v10 = sub_9D1E70((__int64)(v8 + 36), 56, 56, 3);
  sub_E81B30(v10, 0, 0);
  *(_BYTE *)(v10 + 31) &= ~1u;
  *(_BYTE *)(v10 + 30) = a2;
  *(_QWORD *)(v10 + 32) = a3;
  *(_DWORD *)(v10 + 40) = a4;
  *(_DWORD *)(v10 + 44) = v7;
  *(_QWORD *)(v10 + 48) = 0;
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  *(_QWORD *)(v10 + 8) = v11;
  *(_DWORD *)(v10 + 24) = *(_DWORD *)(*(_QWORD *)(a1 + 288) + 24LL) + 1;
  **(_QWORD **)(a1 + 288) = v10;
  *(_QWORD *)(a1 + 288) = v10;
  *(_QWORD *)(*(_QWORD *)(v11 + 8) + 8LL) = v10;
  result = *(_QWORD *)(v10 + 8);
  if ( *(_BYTE *)(result + 32) < a2 )
    *(_BYTE *)(result + 32) = a2;
  return result;
}
