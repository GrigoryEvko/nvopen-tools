// Function: sub_2FEE6E0
// Address: 0x2fee6e0
//
__int64 __fastcall sub_2FEE6E0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rbx
  int v8; // eax
  unsigned __int64 v10; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v11; // [rsp+8h] [rbp-58h]
  unsigned __int64 v12[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v13; // [rsp+30h] [rbp-30h]

  v10 = a2;
  v11 = a3;
  LOBYTE(v12[0]) = 44;
  v4 = sub_C931B0((__int64 *)&v10, v12, 1u, 0);
  if ( v4 == -1 )
  {
    v6 = v10;
    v7 = v11;
    v8 = 0;
  }
  else
  {
    v5 = v4 + 1;
    v6 = v10;
    if ( v4 + 1 > v11 )
    {
      v7 = v11;
      if ( v4 <= v11 )
        v7 = v4;
      v8 = 0;
    }
    else
    {
      if ( v4 > v11 )
        v4 = v11;
      v7 = v4;
      if ( v11 == v5 )
      {
        v8 = 0;
      }
      else if ( sub_C93C90(v10 + v5, v11 - v5, 0xAu, v12) || (v8 = v12[0], v12[0] != LODWORD(v12[0])) )
      {
        v13 = 1283;
        v12[0] = (unsigned __int64)"invalid pass instance specifier ";
        v12[2] = v10;
        v12[3] = v11;
        sub_C64D30((__int64)v12, 1u);
      }
    }
  }
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 8) = v7;
  *(_DWORD *)(a1 + 16) = v8;
  return a1;
}
