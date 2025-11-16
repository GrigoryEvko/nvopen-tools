// Function: sub_2E11E80
// Address: 0x2e11e80
//
__int64 __fastcall sub_2E11E80(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  unsigned int v4; // r8d
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned __int64 v7; // rax

  sub_2E1DCC0(a1[6], *a1, a1[4], a1[5], a1 + 7);
  v3 = a1[1];
  v4 = 0;
  v5 = a1[6];
  v6 = *(_QWORD *)(*(_QWORD *)(v3 + 56) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
  if ( v6 )
  {
    if ( (v6 & 4) == 0 )
    {
      v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v7 )
      {
        if ( *(_BYTE *)(v3 + 48) )
          v4 = *(unsigned __int8 *)(v7 + 43);
      }
    }
  }
  sub_3507530(v5, a2, v4);
  return sub_2E11C60((__int64)a1, a2, 0);
}
