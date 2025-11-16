// Function: sub_1D00F50
// Address: 0x1d00f50
//
__int64 __fastcall sub_1D00F50(__int64 a1)
{
  unsigned int v1; // r14d
  _QWORD *v2; // r12
  _QWORD *i; // r13
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned int v6; // eax

  v1 = 0;
  v2 = *(_QWORD **)(a1 + 112);
  for ( i = &v2[2 * *(unsigned int *)(a1 + 120)]; i != v2; v2 += 2 )
  {
    if ( (*v2 & 6) == 0 )
    {
      v4 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = v4;
      if ( (*(_BYTE *)(v4 + 236) & 2) == 0 )
      {
        sub_1F01F70(v4);
        v5 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
      }
      v6 = *(_DWORD *)(v4 + 244);
      if ( *(_QWORD *)v5 && *(_WORD *)(*(_QWORD *)v5 + 24LL) == 46 )
        v6 = sub_1D00F50() + 1;
      if ( v1 < v6 )
        v1 = v6;
    }
  }
  return v1;
}
