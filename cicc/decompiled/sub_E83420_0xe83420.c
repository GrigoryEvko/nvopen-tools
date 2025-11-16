// Function: sub_E83420
// Address: 0xe83420
//
__int64 __fastcall sub_E83420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  result = sub_E5CB20(*(_QWORD *)(a1 + 296), a2, a3, a4, a5, a6);
  if ( (*(_BYTE *)(a2 + 8) & 0x20) != 0 )
  {
    result = sub_E5CB20(*(_QWORD *)(a1 + 296), a3, v8, v9, v10, v11);
    *(_BYTE *)(a3 + 8) |= 0x20u;
    *(_WORD *)(a3 + 12) &= ~1u;
    if ( *(char *)(a2 + 12) >= 0 )
    {
LABEL_3:
      if ( (*(_BYTE *)(a2 + 8) & 0x40) == 0 )
        return result;
LABEL_7:
      result = sub_E5CB20(*(_QWORD *)(a1 + 296), a3, v8, v9, v10, v11);
      *(_BYTE *)(a3 + 8) |= 0x60u;
      return result;
    }
  }
  else if ( *(char *)(a2 + 12) >= 0 )
  {
    goto LABEL_3;
  }
  result = sub_E5CB20(*(_QWORD *)(a1 + 296), a3, v8, v9, v10, v11);
  *(_WORD *)(a3 + 12) |= 0x80u;
  if ( (*(_BYTE *)(a2 + 8) & 0x40) != 0 )
    goto LABEL_7;
  return result;
}
