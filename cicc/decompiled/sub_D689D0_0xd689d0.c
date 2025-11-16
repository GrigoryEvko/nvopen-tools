// Function: sub_D689D0
// Address: 0xd689d0
//
_QWORD *__fastcall sub_D689D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  int v7; // esi
  unsigned int v8; // eax
  __int64 v9; // rdi
  _QWORD *result; // rax
  unsigned int v11; // esi

  v6 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( *(_DWORD *)(a1 + 76) == v6 )
  {
    v11 = (v6 >> 1) + v6;
    if ( v11 < 2 )
      v11 = 2;
    *(_DWORD *)(a1 + 76) = v11;
    sub_BD2A80(a1, v11, 1);
    v6 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  }
  v7 = (v6 + 1) & 0x7FFFFFF;
  v8 = v7 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  v9 = *(_QWORD *)(a1 - 8) + 32LL * (unsigned int)(v7 - 1);
  *(_DWORD *)(a1 + 4) = v8;
  sub_AC2B30(v9, a2);
  result = (_QWORD *)(*(_QWORD *)(a1 - 8)
                    + 32LL * *(unsigned int *)(a1 + 76)
                    + 8LL * ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) - 1));
  *result = a3;
  return result;
}
