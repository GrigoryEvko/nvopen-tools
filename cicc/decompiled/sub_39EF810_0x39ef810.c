// Function: sub_39EF810
// Address: 0x39ef810
//
void __fastcall sub_39EF810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // r15
  unsigned int v9; // eax

  v5 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v5 && (v8 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v5 - 32)) != 0 )
  {
    if ( sub_39EF7F0(a1) )
      sub_16BD130("Unterminated .bundle_lock when changing a section", 1u);
    v6 = *(_QWORD *)(a1 + 264);
    v9 = *(_DWORD *)(v6 + 480);
    if ( v9 && (*(_BYTE *)(v8 + 44) & 2) != 0 && v9 > *(_DWORD *)(v8 + 24) )
      *(_DWORD *)(v8 + 24) = v9;
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 264);
  }
  v7 = *(_QWORD *)(a2 + 184);
  if ( v7 )
    sub_390D5F0(v6, v7, 0);
  sub_38D58D0(a1, a2, a3);
  sub_390D5F0(v6, *(_QWORD *)(a2 + 8), 0);
}
