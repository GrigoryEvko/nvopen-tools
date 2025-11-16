// Function: sub_E85210
// Address: 0xe85210
//
__int64 __fastcall sub_E85210(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 result; // rax

  if ( (*(_BYTE *)(a2 + 8) & 2) == 0 || (*(_BYTE *)(a2 + 9) & 8) != 0 )
  {
    v5 = *(_QWORD **)(a1 + 8);
    v6 = v5[36];
    v5[46] += 208LL;
    v7 = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5[37] >= (unsigned __int64)(v7 + 208) && v6 )
      v5[36] = v7 + 208;
    else
      v7 = sub_9D1E70((__int64)(v5 + 36), 208, 208, 3);
    sub_E81B30(v7, 1, 0);
    *(_BYTE *)(v7 + 30) = 0;
    *(_QWORD *)(v7 + 40) = v7 + 64;
    *(_QWORD *)(v7 + 96) = v7 + 112;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 56) = 32;
    *(_QWORD *)(v7 + 104) = 0x400000000LL;
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
    *(_QWORD *)(v7 + 8) = v8;
    *(_DWORD *)(v7 + 24) = *(_DWORD *)(*(_QWORD *)(a1 + 288) + 24LL) + 1;
    **(_QWORD **)(a1 + 288) = v7;
    *(_QWORD *)(a1 + 288) = v7;
    *(_QWORD *)(*(_QWORD *)(v8 + 8) + 8LL) = v7;
  }
  result = sub_E8DC70(a1, a2, a3);
  *(_WORD *)(a2 + 12) &= 0xFFF8u;
  return result;
}
