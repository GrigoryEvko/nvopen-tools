// Function: sub_E8B980
// Address: 0xe8b980
//
__int64 __fastcall sub_E8B980(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx

  v5 = (_QWORD *)a1[1];
  v6 = v5[36];
  v5[46] += 240LL;
  v7 = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5[37] >= (unsigned __int64)(v7 + 240) && v6 )
    v5[36] = v7 + 240;
  else
    v7 = sub_9D1E70((__int64)(v5 + 36), 240, 240, 3);
  sub_E81B30(v7, 4, 1);
  *(_BYTE *)(v7 + 30) = 0;
  *(_QWORD *)(v7 + 40) = v7 + 64;
  *(_QWORD *)(v7 + 72) = v7 + 88;
  *(_QWORD *)(v7 + 32) = 0;
  *(_QWORD *)(v7 + 48) = 0;
  *(_QWORD *)(v7 + 56) = 8;
  *(_QWORD *)(v7 + 80) = 0x100000000LL;
  *(_QWORD *)(v7 + 112) = *(_QWORD *)a2;
  *(_QWORD *)(v7 + 120) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(v7 + 128) = v7 + 144;
  *(_QWORD *)(v7 + 136) = 0x600000000LL;
  if ( *(_DWORD *)(a2 + 24) )
    sub_E8A430(v7 + 128, a2 + 16, v8, v9, v10, v11);
  *(_QWORD *)(v7 + 32) = a3;
  v12 = *(_QWORD *)(a1[36] + 8LL);
  *(_QWORD *)(v7 + 8) = v12;
  *(_DWORD *)(v7 + 24) = *(_DWORD *)(a1[36] + 24LL) + 1;
  *(_QWORD *)a1[36] = v7;
  a1[36] = v7;
  *(_QWORD *)(*(_QWORD *)(v12 + 8) + 8LL) = v7;
  return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1[37] + 16LL) + 24LL))(
           *(_QWORD *)(a1[37] + 16LL),
           a2,
           v7 + 40,
           v7 + 72,
           a3);
}
