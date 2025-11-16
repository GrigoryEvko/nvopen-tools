// Function: sub_630970
// Address: 0x630970
//
__int64 __fastcall sub_630970(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rcx
  int v6; // eax
  __int64 v7; // [rsp+0h] [rbp-20h]
  __int64 v8; // [rsp+8h] [rbp-18h]

  if ( dword_4D0488C
    || (result = (__int64)&word_4D04898, word_4D04898)
    && (result = (__int64)&qword_4F077B4, (_DWORD)qword_4F077B4)
    && (result = (__int64)&qword_4F077A0, qword_4F077A0 > 0x765Bu)
    && (v7 = a3, result = sub_729F80(dword_4F063F8), a3 = v7, (_DWORD)result) )
  {
    result = *(_QWORD *)(a2 + 8);
    if ( !result )
      return result;
    if ( *(_BYTE *)(result + 48) != 5 )
      return result;
    result = *(_QWORD *)(result + 56);
    if ( !result || (*(_BYTE *)(result + 193) & 7) != 0 )
      return result;
  }
  else if ( *(char *)(a2 + 42) >= 0 )
  {
    return result;
  }
  if ( (*(_BYTE *)(a1 + 193) & 0x10) == 0 )
  {
    result = *(unsigned __int8 *)(a1 + 195);
    if ( (*(_BYTE *)(a1 + 195) & 3) == 1 )
      goto LABEL_17;
    goto LABEL_10;
  }
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  if ( (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) > 2u || (*(_DWORD *)(v5 + 176) & 0x11000) != 0x1000 )
  {
LABEL_10:
    if ( (*(_BYTE *)(a1 + 206) & 8) == 0 )
    {
      if ( dword_4D0488C
        || word_4D04898
        && (_DWORD)qword_4F077B4
        && qword_4F077A0 > 0x765Bu
        && (v8 = a3, v6 = sub_729F80(dword_4F063F8), a3 = v8, v6) )
      {
        v4 = 2782;
      }
      else
      {
        v4 = 2419;
      }
      return sub_6851C0(v4, a3);
    }
  }
  result = *(unsigned __int8 *)(a1 + 195);
LABEL_17:
  if ( (result & 8) == 0 )
    *(_BYTE *)(a1 + 193) &= ~2u;
  return result;
}
