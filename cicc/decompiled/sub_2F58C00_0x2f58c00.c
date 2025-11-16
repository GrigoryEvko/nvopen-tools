// Function: sub_2F58C00
// Address: 0x2f58c00
//
void __fastcall sub_2F58C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  __int64 *i; // r13
  __int64 v8; // rdx

  v6 = *(__int64 **)(a1 + 28984);
  for ( i = &v6[*(unsigned int *)(a1 + 28992)]; i != v6; ++v6 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL);
    if ( *(_DWORD *)(v8 + 4LL * (*(_DWORD *)(*v6 + 112) & 0x7FFFFFFF)) )
      sub_2F58670(a1, *v6, v8, a4, a5, a6);
  }
}
