// Function: sub_20566E0
// Address: 0x20566e0
//
__int64 __fastcall sub_20566E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rsi
  unsigned __int8 *v9; // rsi
  int v10; // eax
  __int64 v11; // rsi

  v4 = *(_QWORD *)(a1 + 8);
  if ( a2 + 80 != v4 )
  {
    v5 = v4 - (a2 + 80);
    v6 = 0xCCCCCCCCCCCCCCCDLL * (v5 >> 4);
    if ( v5 > 0 )
    {
      v7 = a2 + 136;
      do
      {
        v8 = *(_QWORD *)(v7 - 80);
        *(_DWORD *)(v7 - 136) = *(_DWORD *)(v7 - 56);
        *(_QWORD *)(v7 - 128) = *(_QWORD *)(v7 - 48);
        *(_QWORD *)(v7 - 120) = *(_QWORD *)(v7 - 40);
        *(_QWORD *)(v7 - 112) = *(_QWORD *)(v7 - 32);
        *(_QWORD *)(v7 - 104) = *(_QWORD *)(v7 - 24);
        *(_QWORD *)(v7 - 96) = *(_QWORD *)(v7 - 16);
        *(_QWORD *)(v7 - 88) = *(_QWORD *)(v7 - 8);
        if ( v8 )
          sub_161E7C0(v7 - 80, v8);
        v9 = *(unsigned __int8 **)v7;
        *(_QWORD *)(v7 - 80) = *(_QWORD *)v7;
        if ( v9 )
        {
          sub_1623210(v7, v9, v7 - 80);
          *(_QWORD *)v7 = 0;
        }
        v10 = *(_DWORD *)(v7 + 8);
        v7 += 80;
        *(_DWORD *)(v7 - 152) = v10;
        *(_DWORD *)(v7 - 144) = *(_DWORD *)(v7 - 64);
        *(_DWORD *)(v7 - 140) = *(_DWORD *)(v7 - 60);
        --v6;
      }
      while ( v6 );
      v4 = *(_QWORD *)(a1 + 8);
    }
  }
  *(_QWORD *)(a1 + 8) = v4 - 80;
  v11 = *(_QWORD *)(v4 - 24);
  if ( v11 )
    sub_161E7C0(v4 - 24, v11);
  return a2;
}
