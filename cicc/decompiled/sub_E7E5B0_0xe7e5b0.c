// Function: sub_E7E5B0
// Address: 0xe7e5b0
//
__int64 __fastcall sub_E7E5B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  __int64 v8; // r12
  __int64 v9; // r14
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9

  v6 = a3;
  v8 = *(_QWORD *)(a1 + 288);
  v9 = *(_QWORD *)(a1 + 296);
  if ( v8 )
  {
    if ( sub_E7E4B0(a1) )
      sub_C64ED0("Unterminated .bundle_lock when changing a section", 1u);
    v10 = *(unsigned int *)(v9 + 368);
    if ( (_DWORD)v10 )
    {
      a3 = *(_QWORD *)(v8 + 8);
      if ( (*(_BYTE *)(a3 + 48) & 2) != 0 )
      {
        _BitScanReverse64(&v10, v10);
        LODWORD(v10) = v10 ^ 0x3F;
        a4 = (unsigned int)(63 - v10);
        if ( (unsigned __int8)(63 - v10) > *(_BYTE *)(a3 + 32) )
          *(_BYTE *)(a3 + 32) = 63 - v10;
      }
    }
  }
  v11 = *(_QWORD *)(a2 + 168) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)(a2 + 168) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    sub_E5CB20(v9, v11, a3, a4, a5, a6);
  if ( (*(_BYTE *)(a2 + 154) & 0x20) != 0 )
    *(_BYTE *)(sub_E7DDE0(a1) + 201) = 1;
  sub_E8CEC0(a1, a2, v6);
  return sub_E5CB20(v9, *(_QWORD *)(a2 + 16), v12, v13, v14, v15);
}
