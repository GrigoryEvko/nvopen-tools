// Function: sub_399F6B0
// Address: 0x399f6b0
//
void __fastcall sub_399F6B0(_DWORD *a1, __int64 a2)
{
  __int64 *v2; // rbx
  unsigned int v3; // r13d
  unsigned int i; // r15d
  unsigned int v5; // edx
  unsigned int v6; // esi
  __int64 v7; // rsi

  v2 = (__int64 *)a2;
  a1[19] = 3;
  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 <= 0x40 )
  {
    if ( !v3 )
      return;
  }
  else
  {
    v2 = *(__int64 **)a2;
  }
  for ( i = 0; i < v3; i += 64 )
  {
    v7 = *v2++;
    sub_399F670(a1, v7);
    if ( !i && v3 <= 0x40 )
      break;
    sub_399F600((__int64)a1);
    v5 = i;
    v6 = v3 - i;
    if ( v3 - i > 0x40 )
      v6 = 64;
    sub_399EA60((__int64)a1, v6, v5);
  }
}
