// Function: sub_10B13E0
// Address: 0x10b13e0
//
void __fastcall sub_10B13E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(a2 - 8) + 32LL * a3;
  else
    v6 = a2 + 32 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v7 = *(_QWORD *)v6;
  if ( *(_QWORD *)v6 )
  {
    v8 = *(_QWORD *)(v6 + 8);
    **(_QWORD **)(v6 + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v6 + 16);
  }
  *(_QWORD *)v6 = a4;
  if ( a4 )
  {
    v9 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(v6 + 8) = v9;
    if ( v9 )
    {
      a5 = v6 + 8;
      *(_QWORD *)(v9 + 16) = v6 + 8;
    }
    *(_QWORD *)(v6 + 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = v6;
  }
  if ( *(_BYTE *)v7 > 0x1Cu )
  {
    v10 = *(_QWORD *)(a1 + 40);
    v17[0] = v7;
    v11 = v10 + 2096;
    sub_10B0DA0(v11, v17, v6, a4, a5, a6);
    v16 = *(_QWORD *)(v7 + 16);
    if ( v16 )
    {
      if ( !*(_QWORD *)(v16 + 8) )
      {
        v17[0] = *(_QWORD *)(v16 + 24);
        sub_10B0DA0(v11, v17, v12, v13, v14, v15);
      }
    }
  }
}
