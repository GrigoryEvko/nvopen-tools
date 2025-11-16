// Function: sub_193F190
// Address: 0x193f190
//
__int64 __fastcall sub_193F190(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 v5; // r12
  _QWORD *v7; // rdi
  __int64 v8; // rdx
  unsigned __int8 v9; // cl
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rax

  v3 = *(_BYTE *)(a1 + 16);
  if ( v3 <= 0x17u )
    return 0;
  switch ( v3 )
  {
    case '%':
      goto LABEL_8;
    case '8':
      if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 2 )
        return 0;
LABEL_8:
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v7 = *(_QWORD **)(a1 - 8);
      else
        v7 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v8 = *v7;
      v9 = *(_BYTE *)(*v7 + 16LL);
      if ( v9 == 77 && (v11 = **(_QWORD **)(a2 + 32), v11 == *(_QWORD *)(v8 + 40)) )
      {
        v13 = v7[3];
        v5 = *v7;
        if ( *(_BYTE *)(v13 + 16) <= 0x17u )
          return v5;
        v12 = *(_QWORD *)(v13 + 40);
      }
      else
      {
        if ( v3 == 56 )
          return 0;
        v5 = v7[3];
        if ( *(_BYTE *)(v5 + 16) != 77 )
          return 0;
        v10 = *(__int64 **)(a2 + 32);
        v11 = *v10;
        if ( *v10 != *(_QWORD *)(v5 + 40) )
          return 0;
        if ( v9 <= 0x17u )
          return v5;
        v12 = *(_QWORD *)(v8 + 40);
      }
      if ( sub_15CC890(a3, v12, v11) )
        return v5;
      return 0;
    case '#':
      goto LABEL_8;
  }
  return 0;
}
