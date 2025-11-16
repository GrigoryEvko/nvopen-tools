// Function: sub_16A5D70
// Address: 0x16a5d70
//
__int64 __fastcall sub_16A5D70(__int64 a1, __int64 *a2, unsigned int a3)
{
  unsigned int v3; // eax

  v3 = *((_DWORD *)a2 + 2);
  if ( v3 < a3 )
  {
    sub_16A5B10(a1, a2, a3);
    return a1;
  }
  else if ( v3 > a3 )
  {
    sub_16A5A50(a1, a2, a3);
    return a1;
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v3;
    if ( v3 > 0x40 )
      sub_16A4FD0(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *a2;
    return a1;
  }
}
