// Function: sub_8841F0
// Address: 0x8841f0
//
__int64 __fastcall sub_8841F0(__int64 a1, int a2, __int64 a3, _DWORD *a4)
{
  bool v5; // zf
  __int64 v6; // r13
  _BOOL4 v7; // ecx
  __int64 result; // rax
  __int64 v9; // [rsp-10h] [rbp-30h]

  v5 = a4 == 0;
  v6 = *(_QWORD *)(a1 + 24);
  v7 = a4 == 0;
  if ( v5 )
  {
    result = sub_87DC80(a1, a2, a3, v7);
    if ( (_DWORD)result )
      return result;
  }
  else
  {
    *a4 = 0;
    result = sub_87DC80(a1, a2, a3, v7);
    if ( (_DWORD)result )
    {
      *a4 = 1;
      return result;
    }
  }
  if ( (*(_BYTE *)(a1 + 18) & 1) == 0 )
  {
    if ( (*(_DWORD *)(a1 + 16) & 0x20001) != 0x20001 || (result = sub_8D2870(*(_QWORD *)(a1 + 32)), !(_DWORD)result) )
    {
      result = sub_884000(v6, 1);
      if ( !(_DWORD)result )
      {
        sub_87D9B0(v6, 0, 0, (FILE *)(a1 + 8), a1, 3, 0, a4);
        return v9;
      }
    }
  }
  return result;
}
