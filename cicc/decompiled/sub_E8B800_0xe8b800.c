// Function: sub_E8B800
// Address: 0xe8b800
//
__int64 __fastcall sub_E8B800(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rax
  __int64 v4; // rdx
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 result; // rax
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(__int64 (**)(void))(*(_QWORD *)a1 + 80LL);
  if ( (char *)v3 == (char *)sub_E8A180 )
  {
    v4 = 0;
    if ( *(_BYTE *)(a1 + 276) )
      v4 = *(_QWORD *)(a1 + 296);
  }
  else
  {
    v4 = v3();
  }
  if ( sub_E81930(a2, v10, v4) )
    return sub_E990E0(a1, v10[0]);
  v5 = *(_QWORD **)(a1 + 8);
  v6 = v5[36];
  v5[46] += 104LL;
  v7 = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5[37] >= (unsigned __int64)(v7 + 104) && v6 )
    v5[36] = v7 + 104;
  else
    v7 = sub_9D1E70((__int64)(v5 + 36), 104, 104, 3);
  sub_E81B30(v7, 8, 0);
  *(_BYTE *)(v7 + 30) = 0;
  *(_QWORD *)(v7 + 40) = v7 + 64;
  *(_QWORD *)(v7 + 32) = 0;
  *(_QWORD *)(v7 + 56) = 8;
  *(_QWORD *)(v7 + 72) = v7 + 88;
  *(_QWORD *)(v7 + 80) = 0;
  *(_BYTE *)(v7 + 88) = 1;
  *(_QWORD *)(v7 + 96) = a2;
  *(_BYTE *)(v7 + 64) = 0;
  *(_QWORD *)(v7 + 48) = 1;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  *(_QWORD *)(v7 + 8) = v8;
  *(_DWORD *)(v7 + 24) = *(_DWORD *)(*(_QWORD *)(a1 + 288) + 24LL) + 1;
  **(_QWORD **)(a1 + 288) = v7;
  *(_QWORD *)(a1 + 288) = v7;
  result = *(_QWORD *)(v8 + 8);
  *(_QWORD *)(result + 8) = v7;
  return result;
}
