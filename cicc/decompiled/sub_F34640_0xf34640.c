// Function: sub_F34640
// Address: 0xf34640
//
__int64 __fastcall sub_F34640(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // r13
  unsigned __int8 *v4; // rdi
  unsigned __int16 v6; // dx

  v2 = *(unsigned __int8 **)a1;
  if ( !*(_QWORD *)a1 )
  {
    sub_BD84D0(0, (__int64)a2);
    BUG();
  }
  sub_BD84D0((__int64)(v2 - 24), (__int64)a2);
  if ( (*(v2 - 17) & 0x10) != 0 && (a2[7] & 0x10) == 0 )
    sub_BD6B90(a2, v2 - 24);
  v4 = *(unsigned __int8 **)a1;
  if ( *(_QWORD *)a1 )
    v4 = (unsigned __int8 *)(*(_QWORD *)a1 - 24LL);
  *(_QWORD *)a1 = sub_B43D60(v4);
  *(_WORD *)(a1 + 8) = v6;
  return v6;
}
