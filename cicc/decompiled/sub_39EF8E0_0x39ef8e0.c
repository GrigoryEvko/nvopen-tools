// Function: sub_39EF8E0
// Address: 0x39ef8e0
//
__int64 __fastcall sub_39EF8E0(__int64 a1, char a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // r14
  __int64 v9; // rax

  v2 = 0;
  v3 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v3 )
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v3 - 32);
  if ( !*(_DWORD *)(*(_QWORD *)(a1 + 264) + 480LL) )
    sub_16BD130(".bundle_lock forbidden when bundling is disabled", 1u);
  if ( sub_39EF7F0(a1) )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 264) + 484LL) & 1) == 0 )
      return sub_38D7880(v2, 1 - ((unsigned int)(a2 == 0) - 1));
  }
  else
  {
    *(_BYTE *)(v2 + 44) |= 1u;
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 264) + 484LL) & 1) == 0 )
      return sub_38D7880(v2, 1 - ((unsigned int)(a2 == 0) - 1));
  }
  if ( !sub_39EF7F0(a1) )
  {
    v5 = sub_22077B0(0xE0u);
    v8 = v5;
    if ( v5 )
    {
      sub_38CF760(v5, 1, 0, 0);
      *(_QWORD *)(v8 + 56) = 0;
      *(_WORD *)(v8 + 48) = 0;
      *(_QWORD *)(v8 + 64) = v8 + 80;
      *(_QWORD *)(v8 + 72) = 0x2000000000LL;
      *(_QWORD *)(v8 + 112) = v8 + 128;
      *(_QWORD *)(v8 + 120) = 0x400000000LL;
    }
    v9 = *(unsigned int *)(a1 + 336);
    if ( (unsigned int)v9 >= *(_DWORD *)(a1 + 340) )
    {
      sub_16CD150(a1 + 328, (const void *)(a1 + 344), 0, 8, v6, v7);
      v9 = *(unsigned int *)(a1 + 336);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 328) + 8 * v9) = v8;
    ++*(_DWORD *)(a1 + 336);
  }
  return sub_38D7880(v2, 1 - ((unsigned int)(a2 == 0) - 1));
}
