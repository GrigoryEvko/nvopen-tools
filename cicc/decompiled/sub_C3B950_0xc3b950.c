// Function: sub_C3B950
// Address: 0xc3b950
//
__int64 __fastcall sub_C3B950(__int64 a1, __int64 a2, char a3)
{
  char *v4; // rax
  char v5; // al
  __int64 result; // rax
  char v7; // dl
  char v8; // cl
  int v9; // r13d

  v4 = (char *)sub_C94E20(qword_4F863F0);
  if ( v4 )
    v5 = *v4;
  else
    v5 = qword_4F863F0[2];
  if ( v5 && *(_DWORD **)a1 == dword_3F657C0 )
    return sub_C3B7D0((__int64 *)a1, (__int64 *)a2, a3);
  *(_BYTE *)(a1 + 20) = (*(_BYTE *)(a1 + 20) ^ *(_BYTE *)(a2 + 20)) & 8 | *(_BYTE *)(a1 + 20) & 0xF7;
  result = sub_C392E0((_BYTE *)a1, (_BYTE *)a2);
  v7 = *(_BYTE *)(a1 + 20);
  v8 = v7 & 7;
  if ( (v7 & 7) == 3 )
  {
    if ( *(_DWORD *)(*(_QWORD *)a1 + 20LL) != 2 )
      return result;
    v7 &= ~8u;
    *(_BYTE *)(a1 + 20) = v7;
    v8 = v7 & 7;
  }
  if ( (v7 & 6) != 0 && v8 != 3 )
  {
    v9 = sub_C3A020((__int64 *)a1, a2);
    result = sub_C36450(a1, a3, v9);
    if ( v9 )
      return (unsigned int)result | 0x10;
  }
  return result;
}
