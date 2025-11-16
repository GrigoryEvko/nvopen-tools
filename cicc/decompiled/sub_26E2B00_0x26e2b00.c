// Function: sub_26E2B00
// Address: 0x26e2b00
//
_DWORD *__fastcall sub_26E2B00(_QWORD *a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  _DWORD *v5; // r8
  _DWORD *v6; // rax
  unsigned __int64 v7; // rcx

  v5 = *(_DWORD **)(*a1 + 8 * a2);
  if ( v5 )
  {
    v6 = *(_DWORD **)v5;
    v7 = *(_QWORD *)(*(_QWORD *)v5 + 24LL);
    while ( v7 != a4 || *a3 != v6[2] || a3[1] != v6[3] )
    {
      if ( !*(_QWORD *)v6 )
        return 0;
      v7 = *(_QWORD *)(*(_QWORD *)v6 + 24LL);
      v5 = v6;
      if ( a2 != v7 % a1[1] )
        return 0;
      v6 = *(_DWORD **)v6;
    }
  }
  return v5;
}
