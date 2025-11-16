// Function: sub_14ACAF0
// Address: 0x14acaf0
//
__int64 __fastcall sub_14ACAF0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rbx
  unsigned __int8 v7; // al
  __int64 *v9; // rbx
  __int64 v10; // rdx
  _QWORD *v11; // rax

  while ( 1 )
  {
    v6 = sub_1649C60(a1);
    v7 = *(_BYTE *)(v6 + 16);
    if ( v7 > 0x17u )
    {
      if ( v7 != 56 )
        return 0;
      goto LABEL_6;
    }
    if ( v7 != 5 )
      break;
    if ( *(_WORD *)(v6 + 18) != 32 )
      return 0;
LABEL_6:
    if ( !(unsigned __int8)sub_14ACA50(v6, a3) )
      return 0;
    v9 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
       ? *(__int64 **)(v6 - 8)
       : (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    v10 = v9[6];
    if ( *(_BYTE *)(v10 + 16) != 13 )
      return 0;
    v11 = *(_QWORD **)(v10 + 24);
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
      v11 = (_QWORD *)*v11;
    a1 = *v9;
    a4 += (__int64)v11;
  }
  if ( v7 == 3 && (*(_BYTE *)(v6 + 80) & 1) != 0 && !(unsigned __int8)sub_15E4F60(v6) )
    __asm { jmp     rax }
  return 0;
}
