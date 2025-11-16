// Function: sub_890230
// Address: 0x890230
//
__int64 __fastcall sub_890230(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 result; // rax

  v4 = sub_878CA0();
  *(_QWORD *)(a1 + 192) = v4;
  v5 = v4;
  *(_QWORD *)(v4 + 16) = *(_QWORD *)(a1 + 232);
  *(_QWORD *)(v4 + 24) = *a3;
  *(_BYTE *)(v4 + 40) = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) >> 1) & 7;
  sub_8603B0(v4, -1, *(_DWORD *)(a1 + 76), *(_DWORD *)(a1 + 80));
  ++*(_QWORD *)(a1 + 216);
  v6 = sub_727300();
  v6[3] = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184);
  v6[4] = *a2;
  *v6 = *(_QWORD *)(a1 + 488);
  v7 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 488) = v6;
  if ( v7 )
    *(_QWORD *)(v7 + 32) = v6;
  v8 = qword_4F04C68[0];
  *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 616) = a1;
  result = *(unsigned int *)(v8 + 776LL * (int)dword_4F04C5C);
  *(_DWORD *)(v5 + 8) = result;
  *a3 = v5;
  return result;
}
