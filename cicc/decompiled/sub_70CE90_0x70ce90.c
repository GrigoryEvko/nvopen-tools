// Function: sub_70CE90
// Address: 0x70ce90
//
void __fastcall sub_70CE90(int *a1, char a2, int a3, int a4, _DWORD *a5, _DWORD *a6, _DWORD *a7, __int64 a8)
{
  _BOOL4 v9; // eax
  _DWORD *v10; // [rsp+0h] [rbp-30h]

  if ( a6 )
    *a6 = 0;
  if ( !a4 )
  {
    *a5 = 1;
    return;
  }
  if ( !a3 || qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 17LL) & 4) != 0 )
  {
    if ( a2 == 8 )
    {
      *a5 = 1;
      if ( a6 )
        return;
      goto LABEL_17;
    }
LABEL_15:
    if ( a6 || a2 != 5 )
      return;
LABEL_17:
    sub_684B30((unsigned int)a1, a7);
    return;
  }
  if ( a2 != 8 )
  {
    v10 = a6;
    v9 = sub_67D3C0(a1, a2, a7);
    a6 = v10;
    if ( !v9 )
      goto LABEL_15;
  }
  if ( a6 )
    *a6 = (_DWORD)a1;
  else
    sub_6851C0((unsigned int)a1, a7);
  sub_72C970(a8);
  *a5 = 0;
}
