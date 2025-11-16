// Function: sub_C3AC70
// Address: 0xc3ac70
//
__int64 __fastcall sub_C3AC70(__int64 a1, __int64 a2, char a3, char a4)
{
  char *v6; // rax
  char v7; // al
  __int64 result; // rax
  int v9; // eax
  unsigned int v10; // [rsp+Ch] [rbp-54h]
  __int64 v11[10]; // [rsp+10h] [rbp-50h] BYREF

  v6 = (char *)sub_C94E20(qword_4F863F0);
  if ( v6 )
    v7 = *v6;
  else
    v7 = qword_4F863F0[2];
  if ( v7 && *(_UNKNOWN **)a1 == &unk_3F657C0 )
  {
    sub_C33EB0(v11, (__int64 *)a2);
    if ( a4 )
      sub_C34440((unsigned __int8 *)v11);
    v10 = sub_C3AAF0((__int64 *)a1, v11, a3);
    sub_C338F0((__int64)v11);
    return v10;
  }
  else
  {
    result = sub_C391B0((_BYTE *)a1, (_BYTE *)a2, a4);
    if ( (_DWORD)result == 2 )
    {
      v9 = sub_C376D0(a1, a2, a4);
      result = sub_C36450(a1, a3, v9);
    }
    if ( (*(_BYTE *)(a1 + 20) & 7) == 3 )
    {
      if ( (*(_BYTE *)(a2 + 20) & 7) != 3
        || ((((unsigned __int8)(*(_BYTE *)(a2 + 20) ^ *(_BYTE *)(a1 + 20)) >> 3) ^ 1) & 1) == a4 )
      {
        *(_BYTE *)(a1 + 20) = (8 * (a3 == 3)) | *(_BYTE *)(a1 + 20) & 0xF7;
      }
      if ( *(_DWORD *)(*(_QWORD *)a1 + 20LL) == 2 )
        *(_BYTE *)(a1 + 20) &= ~8u;
    }
  }
  return result;
}
