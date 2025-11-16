// Function: sub_16BCB40
// Address: 0x16bcb40
//
__int64 *__fastcall sub_16BCB40(__int64 *a1, int a2, __int64 a3)
{
  __int64 v5; // rax

  if ( a2 )
  {
    v5 = sub_22077B0(24);
    if ( v5 )
    {
      *(_DWORD *)(v5 + 8) = a2;
      *(_QWORD *)(v5 + 16) = a3;
      *(_QWORD *)v5 = &unk_49EF1D0;
    }
    *a1 = v5 | 1;
    return a1;
  }
  else
  {
    *a1 = 1;
    return a1;
  }
}
