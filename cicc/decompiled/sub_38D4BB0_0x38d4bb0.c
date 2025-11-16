// Function: sub_38D4BB0
// Address: 0x38d4bb0
//
__int64 __fastcall sub_38D4BB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax

  v2 = sub_38D4B30(a1);
  if ( v2 )
  {
    v3 = v2;
    if ( *(_BYTE *)(v2 + 16) == 1 )
    {
      if ( !*(_BYTE *)(v2 + 17) )
        return v3;
      v4 = *(_QWORD *)(a1 + 264);
      if ( *(_DWORD *)(v4 + 480) )
      {
        if ( (*(_BYTE *)(v4 + 484) & 1) != 0 )
          return v3;
      }
      else if ( !a2 || a2 == *(_QWORD *)(v3 + 56) )
      {
        return v3;
      }
    }
  }
  v5 = sub_22077B0(0xE0u);
  v3 = v5;
  if ( v5 )
  {
    v6 = v5;
    sub_38CF760(v5, 1, 0, 0);
    *(_QWORD *)(v3 + 56) = 0;
    *(_WORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 64) = v3 + 80;
    *(_QWORD *)(v3 + 72) = 0x2000000000LL;
    *(_QWORD *)(v3 + 112) = v3 + 128;
    *(_QWORD *)(v3 + 120) = 0x400000000LL;
  }
  else
  {
    v6 = 0;
  }
  sub_38D4150(a1, v3, 0);
  v7 = *(unsigned int *)(a1 + 120);
  v8 = 0;
  if ( (_DWORD)v7 )
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v7 - 32);
  v9 = *(__int64 **)(a1 + 272);
  v10 = *v9;
  v11 = *(_QWORD *)v3 & 7LL;
  *(_QWORD *)(v3 + 8) = v9;
  v10 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v3 = v10 | v11;
  *(_QWORD *)(v10 + 8) = v6;
  *v9 = *v9 & 7 | v6;
  *(_QWORD *)(v3 + 24) = v8;
  return v3;
}
