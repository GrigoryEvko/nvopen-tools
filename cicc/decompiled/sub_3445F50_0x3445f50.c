// Function: sub_3445F50
// Address: 0x3445f50
//
bool __fastcall sub_3445F50(unsigned __int16 *a1)
{
  unsigned __int16 v1; // ax
  __int64 v3; // rax

  v1 = *a1;
  if ( *a1 )
  {
    if ( v1 == 1 || (unsigned __int16)(v1 - 504) <= 7u )
      BUG();
    v3 = *(_QWORD *)&byte_444C4A0[16 * v1 - 16];
    return v3 && (v3 & 7) == 0;
  }
  else
  {
    return sub_3007260((__int64)a1) && (sub_3007260((__int64)a1) & 7) == 0;
  }
}
