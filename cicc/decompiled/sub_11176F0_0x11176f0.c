// Function: sub_11176F0
// Address: 0x11176f0
//
__int64 __fastcall sub_11176F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // rax
  _QWORD *v5; // rax
  unsigned __int64 v7; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r14
  unsigned __int8 v14; // al
  unsigned __int8 v16; // [rsp-30h] [rbp-30h]

  v4 = *(_QWORD *)(a2 + 40);
  if ( !v4 )
    return 0;
  v5 = (_QWORD *)(v4 + 48);
  v7 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v7 == v5 )
    return 0;
  if ( !v7 )
    BUG();
  if ( *(_BYTE *)(v7 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v7 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v9 = *(_BYTE **)(v7 - 120);
  if ( *v9 != 82 )
    return 0;
  v10 = a4;
  v11 = *((_QWORD *)v9 - 8);
  if ( !v11 || a2 != v11 )
  {
    v12 = *((_QWORD *)v9 - 4);
    if ( a2 != v12 || !v12 )
      return 0;
  }
  if ( (*(_WORD *)(a3 + 2) & 0x3F) != 0x20 )
    return 0;
  v13 = sub_B46EC0(v7 - 24, 1u);
  if ( !sub_AA54C0(v13) )
    return 0;
  v14 = sub_1117660(a1, a2, a3, v13);
  if ( !v14 )
    return 0;
  v16 = v14;
  sub_BD7E80((unsigned __int8 *)a2, *(unsigned __int8 **)(a2 + 32 * v10 - 96), *(__int64 **)(a2 + 40));
  return v16;
}
