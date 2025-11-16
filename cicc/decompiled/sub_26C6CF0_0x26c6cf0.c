// Function: sub_26C6CF0
// Address: 0x26c6cf0
//
__int64 __fastcall sub_26C6CF0(__int64 a1, void (__fastcall ***a2)(unsigned __int64 *, _QWORD, __int64), __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  unsigned __int64 v5; // r12
  __int64 v6; // rdx
  __int64 (__fastcall **v8)(); // rax
  char v9; // [rsp+Fh] [rbp-51h]
  unsigned __int64 v10; // [rsp+10h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-40h]

  v3 = a3 + 48;
  v4 = *(_QWORD *)(a3 + 56);
  if ( v4 == a3 + 48 )
    goto LABEL_12;
  v9 = 0;
  v5 = 0;
  do
  {
    v6 = v4 - 24;
    if ( !v4 )
      v6 = 0;
    (**a2)(&v10, a2, v6);
    if ( (v11 & 1) == 0 )
    {
      v9 = 1;
      if ( v5 < v10 )
        v5 = v10;
    }
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v3 != v4 );
  if ( !v9 )
  {
LABEL_12:
    v8 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v8;
  }
  else
  {
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v5;
  }
  return a1;
}
