// Function: sub_3377160
// Address: 0x3377160
//
__int64 __fastcall sub_3377160(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rdx
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // rsi
  unsigned __int8 *v11; // rsi
  int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rsi

  v3 = *(_QWORD *)(a1 + 8);
  if ( a2 + 96 != v3 )
  {
    v4 = v3 - (a2 + 96);
    v5 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 5);
    if ( v4 > 0 )
    {
      v6 = a2 + 152;
      do
      {
        v7 = *(_QWORD *)(v6 - 96);
        *(_QWORD *)(v6 - 152) = *(_QWORD *)(v6 - 56);
        *(_QWORD *)(v6 - 144) = *(_QWORD *)(v6 - 48);
        *(_QWORD *)(v6 - 136) = *(_QWORD *)(v6 - 40);
        *(_QWORD *)(v6 - 128) = *(_QWORD *)(v6 - 32);
        *(_QWORD *)(v6 - 120) = *(_QWORD *)(v6 - 24);
        *(_QWORD *)(v6 - 112) = *(_QWORD *)(v6 - 16);
        *(_QWORD *)(v6 - 104) = *(_QWORD *)(v6 - 8);
        if ( v7 )
          sub_B91220(v6 - 96, v7);
        v8 = *(unsigned __int8 **)v6;
        *(_QWORD *)(v6 - 96) = *(_QWORD *)v6;
        if ( v8 )
        {
          sub_B976B0(v6, v8, v6 - 96);
          *(_QWORD *)v6 = 0;
        }
        v9 = v6 + 16;
        *(_DWORD *)(v6 - 88) = *(_DWORD *)(v6 + 8);
        if ( v6 + 16 != v6 - 80 )
        {
          v10 = *(_QWORD *)(v6 - 80);
          if ( v10 )
          {
            sub_B91220(v6 - 80, v10);
            v9 = v6 + 16;
          }
          v11 = *(unsigned __int8 **)(v6 + 16);
          *(_QWORD *)(v6 - 80) = v11;
          if ( v11 )
          {
            sub_B976B0(v9, v11, v6 - 80);
            *(_QWORD *)(v6 + 16) = 0;
          }
        }
        v12 = *(_DWORD *)(v6 + 24);
        v6 += 96;
        *(_DWORD *)(v6 - 168) = v12;
        *(_DWORD *)(v6 - 164) = *(_DWORD *)(v6 - 68);
        *(_BYTE *)(v6 - 160) = *(_BYTE *)(v6 - 64);
        --v5;
      }
      while ( v5 );
      v3 = *(_QWORD *)(a1 + 8);
    }
  }
  *(_QWORD *)(a1 + 8) = v3 - 96;
  v13 = *(_QWORD *)(v3 - 24);
  if ( v13 )
    sub_B91220(v3 - 24, v13);
  v14 = *(_QWORD *)(v3 - 40);
  if ( v14 )
    sub_B91220(v3 - 40, v14);
  return a2;
}
