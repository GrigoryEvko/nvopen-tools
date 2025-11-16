// Function: sub_8F0920
// Address: 0x8f0920
//
void __fastcall sub_8F0920(__int64 a1, int a2)
{
  unsigned int v2; // r8d
  int v3; // eax
  int v5; // eax
  int v6; // r13d
  __int64 v7; // r14
  int v8; // edi
  int v9; // ecx
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rax

  v2 = *(_DWORD *)(a1 + 2088);
  if ( v2 )
  {
    v3 = a2 + 31;
    if ( a2 >= 0 )
      v3 = a2;
    v5 = v3 >> 5;
    v6 = v5;
    if ( !v5 )
    {
      if ( (a2 & 0x1F) == 0 )
        return;
      v8 = 0;
      goto LABEL_13;
    }
    v7 = 4LL * (v5 + 2);
    memmove((void *)(v7 + a1), (const void *)(a1 + 8), 4LL * v2);
    memset((void *)(a1 + 8), 0, v7 - 8);
    v2 = v6 + *(_DWORD *)(a1 + 2088);
    *(_DWORD *)(a1 + 2088) = v2;
    if ( (a2 & 0x1F) != 0 )
    {
      if ( v2 )
      {
        v8 = v6;
        if ( v2 > v6 )
        {
LABEL_13:
          v9 = a2 % 32;
          v10 = a1 + 4LL * v6;
          v11 = a1 + 4 * (~v8 + v2 + (__int64)v6) + 4;
          v12 = 0;
          do
          {
            v13 = *(unsigned int *)(v10 + 8);
            v10 += 4;
            v14 = (v13 << v9) | v12;
            *(_DWORD *)(v10 + 4) = v14;
            v12 = HIDWORD(v14);
          }
          while ( v11 != v10 );
          if ( v12 )
          {
            *(_DWORD *)(a1 + 4LL * v2 + 8) = v12;
            *(_DWORD *)(a1 + 2088) = v2 + 1;
          }
        }
      }
    }
  }
}
