// Function: sub_2A398E0
// Address: 0x2a398e0
//
void __fastcall sub_2A398E0(__int64 **a1, unsigned __int8 *a2)
{
  __int64 v2; // rax

  if ( *a2 == 62 )
  {
    sub_2A38DB0(a1, a2);
  }
  else if ( *a2 == 85 )
  {
    v2 = *((_QWORD *)a2 - 4);
    if ( v2 && !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *((_QWORD *)a2 + 10) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      sub_2A391C0(a1, (__int64)a2);
    else
      sub_2A39750((__int64)a1, (__int64)a2);
  }
  else
  {
    sub_2A37680(a1, (__int64)a2);
  }
}
