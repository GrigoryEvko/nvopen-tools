// Function: sub_FDE760
// Address: 0xfde760
//
__int64 __fastcall sub_FDE760(__int64 a1, __int64 a2)
{
  signed __int64 v3; // rdi
  __int16 v5; // r13
  __int16 v6; // bx
  __int16 v7; // dx

  v3 = *(_QWORD *)a1;
  if ( !v3 )
    return a1;
  if ( !*(_QWORD *)a2 )
  {
    *(_QWORD *)a1 = -1;
    *(_WORD *)(a1 + 8) = 0x3FFF;
    return a1;
  }
  v5 = *(_WORD *)(a2 + 8);
  v6 = *(_WORD *)(a1 + 8);
  *(_QWORD *)a1 = sub_F04200(v3, *(_QWORD *)a2);
  *(_WORD *)(a1 + 8) = v7;
  sub_D78C90(a1, (__int16)(v6 - v5));
  return a1;
}
