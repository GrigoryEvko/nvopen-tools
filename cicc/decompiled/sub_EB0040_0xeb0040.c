// Function: sub_EB0040
// Address: 0xeb0040
//
__int64 __fastcall sub_EB0040(__int64 a1, __int64 a2, __int64 a3)
{
  char *v4; // rax
  _QWORD v5[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v6; // [rsp+10h] [rbp-50h]
  __int64 v7; // [rsp+18h] [rbp-48h]
  __int16 v8; // [rsp+20h] [rbp-40h]
  _QWORD v9[4]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v10; // [rsp+50h] [rbp-10h]

  if ( **(_DWORD **)(a1 + 48) == 9 )
  {
    if ( *(_QWORD *)(a1 + 376) != *(_QWORD *)(a1 + 368) )
    {
      sub_EAFEB0(a1);
      return 0;
    }
    v6 = a2;
    v8 = 1283;
    v7 = a3;
    v5[0] = "unexpected '";
    v9[0] = v5;
    v4 = "' in file, no current macro definition";
  }
  else
  {
    v6 = a2;
    v5[0] = "unexpected token in '";
    v9[0] = v5;
    v4 = "' directive";
    v8 = 1283;
    v7 = a3;
  }
  v9[2] = v4;
  v10 = 770;
  return sub_ECE0E0(a1, v9, 0, 0);
}
