// Function: sub_C44AB0
// Address: 0xc44ab0
//
__int64 __fastcall sub_C44AB0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // eax

  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 < a3 )
  {
    sub_C449B0(a1, (const void **)a2, a3);
    return a1;
  }
  else if ( v3 > a3 )
  {
    sub_C44740(a1, (char **)a2, a3);
    return a1;
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v3;
    if ( v3 > 0x40 )
      sub_C43780(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
}
