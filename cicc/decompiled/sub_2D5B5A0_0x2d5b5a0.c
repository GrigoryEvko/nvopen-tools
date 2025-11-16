// Function: sub_2D5B5A0
// Address: 0x2d5b5a0
//
__int64 __fastcall sub_2D5B5A0(__int64 a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  __int64 i; // rax
  _BYTE *v6; // rcx
  char v7; // dl
  __int64 v8; // r12
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  _BYTE *v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // r14d
  __int64 j; // r13
  __int64 *v15; // r12
  char v16; // al

  if ( !byte_5016E68 )
  {
    v3 = *(__int64 (**)())(*(_QWORD *)a2 + 328LL);
    if ( v3 == sub_2D565C0 || ((unsigned __int8 (__fastcall *)(__int64))v3)(a2) )
      return 0;
  }
  if ( (*(_WORD *)(a1 + 2) & 0x3F) != 0x20 )
    return 0;
  for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v6 = *(_BYTE **)(i + 24);
    v7 = *v6;
    if ( *v6 <= 0x1Cu || v7 != 31 && (v7 != 86 || a1 != *((_QWORD *)v6 - 12)) )
      return 0;
  }
  v8 = *(_QWORD *)(a1 + 40);
  v9 = sub_AA54C0(v8);
  if ( !v9 )
    return 0;
  v10 = sub_986580(v9);
  if ( *(_BYTE *)v10 != 31 )
    return 0;
  if ( (*(_DWORD *)(v10 + 4) & 0x7FFFFFF) != 3 )
    return 0;
  v11 = *(_BYTE **)(v10 - 96);
  if ( !v11 )
    return 0;
  if ( !*(_QWORD *)(v10 - 32) )
    return 0;
  v12 = *(_QWORD *)(v10 - 64);
  if ( !v12 )
    return 0;
  if ( v8 != v12 )
    return 0;
  if ( *v11 != 82 )
    return 0;
  if ( *(_QWORD *)(a1 - 64) != *((_QWORD *)v11 - 8) )
    return 0;
  if ( *(_QWORD *)(a1 - 32) != *((_QWORD *)v11 - 4) )
    return 0;
  v13 = sub_B53900((__int64)v11);
  if ( ((v13 - 38) & 0xFFFFFFFD) != 0 )
    return 0;
  for ( j = *(_QWORD *)(a1 + 16); j; j = *(_QWORD *)(j + 8) )
  {
    v15 = *(__int64 **)(j + 24);
    v16 = *(_BYTE *)v15;
    if ( *(_BYTE *)v15 <= 0x1Cu )
LABEL_33:
      BUG();
    if ( v16 == 31 )
    {
      sub_B4CC70(*(_QWORD *)(j + 24));
    }
    else
    {
      if ( v16 != 86 )
        goto LABEL_33;
      sub_BD28A0(v15 - 8, v15 - 4);
      sub_B47280((__int64)v15);
    }
  }
  *(_WORD *)(a1 + 2) = sub_B52F50(v13) | *(_WORD *)(a1 + 2) & 0xFFC0;
  return 1;
}
