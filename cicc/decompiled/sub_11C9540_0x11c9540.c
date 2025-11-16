// Function: sub_11C9540
// Address: 0x11c9540
//
void __fastcall sub_11C9540(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned int v5; // r13d
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rsi
  char v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // [rsp-58h] [rbp-58h]
  __int64 v16; // [rsp-50h] [rbp-50h]
  _QWORD v17[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_QWORD *)(a1 + 104) )
  {
    if ( !(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL) >> 8) && ((*(_WORD *)(a1 + 2) >> 4) & 0x3BF) == 0 )
    {
      v2 = *(_QWORD *)(a1 + 40);
      v5 = (unsigned int)sub_BAA2D0(v2);
      if ( v5 )
      {
        v6 = v2 + 312;
        if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        {
          sub_B2C6D0(a1, a2, v3, v4);
          v7 = *(_QWORD *)(a1 + 96);
          v16 = v7 + 40LL * *(_QWORD *)(a1 + 104);
          if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
          {
            sub_B2C6D0(a1, a2, v13, v14);
            v7 = *(_QWORD *)(a1 + 96);
          }
        }
        else
        {
          v7 = *(_QWORD *)(a1 + 96);
          v16 = v7 + 40LL * *(_QWORD *)(a1 + 104);
        }
        for ( ; v7 != v16; v7 += 40 )
        {
          v8 = *(_QWORD *)(v7 + 8);
          if ( (*(_BYTE *)(v8 + 8) & 0xFD) == 0xC )
          {
            v15 = *(_QWORD *)(v7 + 8);
            v9 = sub_AE5020(v6, v8);
            v10 = sub_9208B0(v6, v15);
            v17[1] = v11;
            v17[0] = (((unsigned __int64)(v10 + 7) >> 3) + (1LL << v9) - 1) >> v9 << v9;
            if ( (unsigned __int64)sub_CA1930(v17) <= 8 )
            {
              v12 = ((unsigned __int64)sub_CA1930(v17) > 4) + 1;
              if ( v12 > v5 )
                return;
              v5 -= v12;
              sub_B2D3C0(a1, *(_DWORD *)(v7 + 32), 15);
            }
          }
        }
      }
    }
  }
}
