// Function: sub_17C7310
// Address: 0x17c7310
//
__int64 __fastcall sub_17C7310(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v3; // eax
  __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // r15
  __int64 v7; // rax
  __int16 v8; // dx
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  __int64 v16; // [rsp+18h] [rbp-48h]
  __int64 v17[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 - 24);
  v3 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
  if ( v3 )
  {
    v4 = 0;
    v16 = v3 - 1;
    while ( 1 )
    {
      v5 = *(_QWORD *)(v2 + 24 * (v4 - v3));
      v6 = (_QWORD *)v5;
      v7 = sub_1649C60(v5);
      v8 = *(_WORD *)(v7 + 32);
      v17[0] = v7;
      *(_WORD *)(v7 + 32) = v8 & 0xBFC0 | 0x4008;
      v9 = *(_BYTE **)(a1 + 176);
      if ( v9 == *(_BYTE **)(a1 + 184) )
      {
        sub_17C7180(a1 + 168, v9, v17);
      }
      else
      {
        if ( v9 )
        {
          *(_QWORD *)v9 = v7;
          v9 = *(_BYTE **)(a1 + 176);
        }
        *(_QWORD *)(a1 + 176) = v9 + 8;
      }
      v10 = 3LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
      v11 = (_QWORD *)(v5 - v10 * 8);
      if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
      {
        v11 = *(_QWORD **)(v5 - 8);
        v6 = &v11[v10];
      }
      for ( ; v6 != v11; v11 += 3 )
      {
        if ( *v11 )
        {
          v12 = v11[1];
          v13 = v11[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v13 = v12;
          if ( v12 )
            *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
        }
        *v11 = 0;
      }
      if ( v16 == v4 )
        break;
      ++v4;
      v3 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
    }
  }
  return sub_15E55B0(a2);
}
