// Function: sub_FE9650
// Address: 0xfe9650
//
__int64 __fastcall sub_FE9650(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r8
  __int16 v6; // r13
  __int16 v7; // bx
  unsigned __int64 v8; // rdi
  __int16 v9; // dx

  v3 = *(_QWORD *)a1;
  if ( !v3 )
    return a1;
  v4 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
  {
    *(_QWORD *)a1 = 0;
    *(_WORD *)(a1 + 8) = *(_WORD *)(a2 + 8);
    return a1;
  }
  v6 = *(_WORD *)(a1 + 8);
  v7 = *(_WORD *)(a2 + 8);
  if ( v3 > 0xFFFFFFFF || v4 > 0xFFFFFFFF )
  {
    v8 = sub_F04140(v3, *(_QWORD *)a2);
  }
  else
  {
    v8 = v4 * v3;
    v9 = 0;
  }
  *(_QWORD *)a1 = v8;
  *(_WORD *)(a1 + 8) = v9;
  sub_D78C90(a1, (__int16)(v7 + v6));
  return a1;
}
