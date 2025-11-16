// Function: sub_FFF8A0
// Address: 0xfff8a0
//
__int64 __fastcall sub_FFF8A0(int a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // r8
  unsigned int v12; // edi
  __int64 v13; // rax

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v4 = *(_QWORD *)(a2 - 32);
  if ( v4
    && !*(_BYTE *)v4
    && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a2 + 80)
    && (*(_BYTE *)(v4 + 33) & 0x20) != 0
    && a1 == *(_DWORD *)(v4 + 36)
    && ((v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF,
         v8 = *(_QWORD *)(a2 - 32 * v7),
         v9 = *(_QWORD *)(a2 + 32 * (1 - v7)),
         a3 == v8)
     || a3 == v9
     || *(_BYTE *)a3 == 85
     && (v10 = *(_QWORD *)(a3 - 32)) != 0
     && !*(_BYTE *)v10
     && *(_QWORD *)(v10 + 24) == *(_QWORD *)(a3 + 80)
     && (*(_BYTE *)(v10 + 33) & 0x20) != 0
     && ((v11 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)),
          v12 = *(_DWORD *)(v10 + 36),
          v13 = *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))),
          v8 == v11)
      && v9 == v13
      || v8 == v13 && v9 == v11)
     && (a1 == v12 || a1 == (unsigned int)sub_9905C0(v12))) )
  {
    return a2;
  }
  else
  {
    return 0;
  }
}
