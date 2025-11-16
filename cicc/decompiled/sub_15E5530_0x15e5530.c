// Function: sub_15E5530
// Address: 0x15e5530
//
__int64 __fastcall sub_15E5530(__int64 a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx

  v2 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v3 = *(_QWORD **)(a1 - 8);
    v4 = &v3[v2];
  }
  else
  {
    v3 = (_QWORD *)(a1 - v2 * 8);
    v4 = (_QWORD *)a1;
  }
  for ( ; v4 != v3; v3 += 3 )
  {
    if ( *v3 )
    {
      v5 = v3[1];
      v6 = v3[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v6 = v5;
      if ( v5 )
        *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
    }
    *v3 = 0;
  }
  return sub_161FB70(a1);
}
