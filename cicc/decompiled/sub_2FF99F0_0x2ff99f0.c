// Function: sub_2FF99F0
// Address: 0x2ff99f0
//
void __fastcall sub_2FF99F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r12
  __int64 v4; // rax
  signed int v5; // r14d
  unsigned int v6; // eax

  if ( *(_WORD *)(a2 + 68) != 20
    || (v4 = *(_QWORD *)(a2 + 32), v5 = *(_DWORD *)(v4 + 8), v5 > 0)
    && (v6 = sub_2FF8D40(*(_DWORD *)(v4 + 48), a1 + 208), !sub_2FF8810(a1, v5, v6)) )
  {
    v2 = *(_QWORD *)(a2 + 32);
    for ( i = v2 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF); i != v2; v2 += 40 )
    {
      if ( *(_BYTE *)v2 == 12 || !*(_BYTE *)v2 && (*(_BYTE *)(v2 + 3) & 0x10) != 0 && *(int *)(v2 + 8) > 0 )
        sub_2FF97C0(a1, v2, a1 + 208);
    }
  }
}
