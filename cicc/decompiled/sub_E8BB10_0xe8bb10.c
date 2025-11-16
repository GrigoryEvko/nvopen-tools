// Function: sub_E8BB10
// Address: 0xe8bb10
//
__int64 __fastcall sub_E8BB10(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  char v4; // al
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx

  v2 = a1[36];
  if ( *(_BYTE *)(v2 + 28) != 1
    || (v4 = *(_BYTE *)(v2 + 29), (v4 & 1) != 0)
    && ((v4 & 4) != 0 || *(_DWORD *)(a1[37] + 368LL) || a2 && a2 != *(_QWORD *)(v2 + 32)) )
  {
    v5 = (_QWORD *)a1[1];
    v6 = v5[36];
    v5[46] += 208LL;
    v2 = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5[37] >= (unsigned __int64)(v2 + 208) && v6 )
      v5[36] = v2 + 208;
    else
      v2 = sub_9D1E70((__int64)(v5 + 36), 208, 208, 3);
    sub_E81B30(v2, 1, 0);
    *(_BYTE *)(v2 + 30) = 0;
    *(_QWORD *)(v2 + 40) = v2 + 64;
    *(_QWORD *)(v2 + 96) = v2 + 112;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 56) = 32;
    *(_QWORD *)(v2 + 104) = 0x400000000LL;
    v7 = *(_QWORD *)(a1[36] + 8LL);
    *(_QWORD *)(v2 + 8) = v7;
    *(_DWORD *)(v2 + 24) = *(_DWORD *)(a1[36] + 24LL) + 1;
    *(_QWORD *)a1[36] = v2;
    a1[36] = v2;
    *(_QWORD *)(*(_QWORD *)(v7 + 8) + 8LL) = v2;
  }
  return v2;
}
