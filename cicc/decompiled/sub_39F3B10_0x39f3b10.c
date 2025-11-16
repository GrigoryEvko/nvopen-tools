// Function: sub_39F3B10
// Address: 0x39f3b10
//
__int64 __fastcall sub_39F3B10(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // rax
  __int64 v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h]
  int v9; // [rsp+18h] [rbp-28h]

  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  if ( (unsigned __int8)sub_38CF2C0(a3, (__int64)&v6, 0, 0) )
  {
    if ( v6 )
    {
      if ( !v7 )
      {
        v5 = *(_BYTE **)(v6 + 24);
        if ( (*v5 & 4) == 0 || !**((_QWORD **)v5 - 1) || v8 )
          *(_WORD *)(a2 + 12) |= 0x200u;
      }
    }
  }
  return sub_38D6700(a1, a2, a3);
}
