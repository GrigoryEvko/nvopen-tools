// Function: sub_E8B440
// Address: 0xe8b440
//
__int64 __fastcall sub_E8B440(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rax
  _QWORD *v6; // rdi
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 result; // rax

  v5 = sub_E8A230(a1, a3, a2, a4);
  v6 = *(_QWORD **)(a1 + 8);
  v7 = v5;
  v8 = v6[36];
  v6[46] += 120LL;
  v9 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6[37] >= (unsigned __int64)(v9 + 120) && v8 )
    v6[36] = v9 + 120;
  else
    v9 = sub_9D1E70((__int64)(v6 + 36), 120, 120, 3);
  sub_E81B30(v9, 7, 0);
  *(_BYTE *)(v9 + 30) = 0;
  *(_QWORD *)(v9 + 40) = v9 + 64;
  *(_QWORD *)(v9 + 72) = v9 + 88;
  *(_QWORD *)(v9 + 32) = 0;
  *(_QWORD *)(v9 + 48) = 0;
  *(_QWORD *)(v9 + 56) = 8;
  *(_QWORD *)(v9 + 80) = 0x100000000LL;
  *(_QWORD *)(v9 + 112) = v7;
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  *(_QWORD *)(v9 + 8) = v10;
  *(_DWORD *)(v9 + 24) = *(_DWORD *)(*(_QWORD *)(a1 + 288) + 24LL) + 1;
  **(_QWORD **)(a1 + 288) = v9;
  *(_QWORD *)(a1 + 288) = v9;
  result = *(_QWORD *)(v10 + 8);
  *(_QWORD *)(result + 8) = v9;
  return result;
}
