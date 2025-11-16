// Function: sub_89EF00
// Address: 0x89ef00
//
__int64 __fastcall sub_89EF00(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdx

  v2 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  sub_891F00(a1, a2);
  v3 = dword_4F04C64;
  *(_DWORD *)(a1 + 20) = 1;
  *(_DWORD *)(a1 + 204) = v3;
  *(_DWORD *)(a1 + 44) = (*(_BYTE *)(v2 + 6) & 2) != 0;
  *(_QWORD *)(a1 + 232) = *(_QWORD *)(v2 + 184);
  *(_QWORD *)(a1 + 240) = *(_QWORD *)(v2 + 208);
  *(_QWORD *)(a1 + 336) = sub_727340();
  result = sub_878CA0();
  v5 = *(_QWORD *)(a1 + 232);
  *(_QWORD *)(a1 + 192) = result;
  *(_QWORD *)(result + 16) = v5;
  *(_BYTE *)(result + 40) = (*(_BYTE *)(v2 + 9) >> 1) & 7;
  return result;
}
