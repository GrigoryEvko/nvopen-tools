// Function: sub_C44B10
// Address: 0xc44b10
//
__int64 __fastcall sub_C44B10(__int64 a1, char **a2, unsigned int a3)
{
  unsigned int v3; // eax

  v3 = *((_DWORD *)a2 + 2);
  if ( v3 < a3 )
  {
    sub_C44830(a1, a2, a3);
    return a1;
  }
  else if ( v3 > a3 )
  {
    sub_C44740(a1, a2, a3);
    return a1;
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v3;
    if ( v3 > 0x40 )
      sub_C43780(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *a2;
    return a1;
  }
}
