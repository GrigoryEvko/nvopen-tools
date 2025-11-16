// Function: sub_20DF590
// Address: 0x20df590
//
void __fastcall sub_20DF590(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v3; // r11
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // ecx
  unsigned int v8; // edi
  unsigned __int64 v9; // rax

  v2 = *(_DWORD *)(a2 + 48);
  v3 = *(_QWORD *)(a1 + 456) + 320LL;
  if ( a2 != v3 )
  {
    v5 = v2;
    while ( 1 )
    {
      if ( v2 )
      {
        v6 = *(_QWORD *)(a1 + 232);
        v7 = *(_DWORD *)(a2 + 176);
        v8 = *(_DWORD *)(v6 + 8 * v5 + 4) + *(_DWORD *)(v6 + 8 * v5);
        if ( v7 )
        {
          v9 = (unsigned int)(1 << v7)
             * ((v8 + (unsigned __int64)(unsigned int)(1 << v7) - 1)
              / (unsigned int)(1 << v7))
             - v8;
          if ( v7 > *(_DWORD *)(*(_QWORD *)(a2 + 56) + 340LL) )
            v8 += 1 << v7;
          v8 += v9;
        }
        *(_DWORD *)(v6 + 8LL * v2) = v8;
        v5 = v2;
      }
      a2 = *(_QWORD *)(a2 + 8);
      if ( v3 == a2 )
        break;
      v2 = *(_DWORD *)(a2 + 48);
    }
  }
}
