// Function: sub_F57330
// Address: 0xf57330
//
__int64 __fastcall sub_F57330(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v15; // [rsp+0h] [rbp-40h]
  unsigned int v16; // [rsp+Ch] [rbp-34h]

  v7 = *(_QWORD *)(a1 + 16);
  v16 = 0;
  v15 = a2 + 16;
  while ( v7 )
  {
    while ( 1 )
    {
      v11 = v7;
      v7 = *(_QWORD *)(v7 + 8);
      v12 = *(_QWORD *)(v11 + 24);
      if ( *(_BYTE *)v12 != 85 )
        break;
      v13 = *(_QWORD *)(v12 - 32);
      if ( !v13
        || *(_BYTE *)v13
        || *(_QWORD *)(v13 + 24) != *(_QWORD *)(v12 + 80)
        || (*(_BYTE *)(v13 + 33) & 0x20) == 0
        || *(_DWORD *)(v13 + 36) != 171 )
      {
        break;
      }
      if ( !v7 )
        return v16;
    }
    sub_B19BE0(a3, a4, v11);
    if ( v8 )
    {
      if ( *(_QWORD *)v11 )
      {
        v9 = *(_QWORD *)(v11 + 8);
        **(_QWORD **)(v11 + 16) = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = *(_QWORD *)(v11 + 16);
      }
      *(_QWORD *)v11 = a2;
      if ( a2 )
      {
        v10 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(v11 + 8) = v10;
        if ( v10 )
          *(_QWORD *)(v10 + 16) = v11 + 8;
        *(_QWORD *)(v11 + 16) = v15;
        *(_QWORD *)(a2 + 16) = v11;
      }
      ++v16;
    }
  }
  return v16;
}
