// Function: sub_3586670
// Address: 0x3586670
//
__int64 __fastcall sub_3586670(__int64 a1, void (__fastcall ***a2)(unsigned __int64 *, _QWORD, __int64), __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned __int64 v5; // r12
  __int64 (__fastcall **v7)(); // rax
  char v8; // [rsp+Fh] [rbp-51h]
  unsigned __int64 v9; // [rsp+10h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-40h]

  v3 = a3 + 48;
  v4 = *(_QWORD *)(a3 + 56);
  if ( v4 == a3 + 48 )
    goto LABEL_16;
  v8 = 0;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      (**a2)(&v9, a2, v4);
      if ( (v10 & 1) == 0 )
      {
        v8 = 1;
        if ( v5 < v9 )
          v5 = v9;
      }
      if ( !v4 )
        BUG();
      if ( (*(_BYTE *)v4 & 4) == 0 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        goto LABEL_10;
    }
    while ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
      v4 = *(_QWORD *)(v4 + 8);
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v3 != v4 );
LABEL_10:
  if ( !v8 )
  {
LABEL_16:
    v7 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v7;
  }
  else
  {
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v5;
  }
  return a1;
}
