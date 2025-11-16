// Function: sub_F57230
// Address: 0xf57230
//
__int64 __fastcall sub_F57230(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v14; // [rsp+0h] [rbp-40h]
  unsigned int v15; // [rsp+Ch] [rbp-34h]

  v7 = *(_QWORD *)(a1 + 16);
  v15 = 0;
  v14 = a2 + 16;
  while ( v7 )
  {
    while ( 1 )
    {
      v10 = v7;
      v7 = *(_QWORD *)(v7 + 8);
      v11 = *(_QWORD *)(v10 + 24);
      if ( *(_BYTE *)v11 != 85 )
        break;
      v12 = *(_QWORD *)(v11 - 32);
      if ( !v12
        || *(_BYTE *)v12
        || *(_QWORD *)(v12 + 24) != *(_QWORD *)(v11 + 80)
        || (*(_BYTE *)(v12 + 33) & 0x20) == 0
        || *(_DWORD *)(v12 + 36) != 171 )
      {
        break;
      }
      if ( !v7 )
        return v15;
    }
    if ( (unsigned __int8)sub_B19ED0(a3, a4, v10) )
    {
      if ( *(_QWORD *)v10 )
      {
        v8 = *(_QWORD *)(v10 + 8);
        **(_QWORD **)(v10 + 16) = v8;
        if ( v8 )
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v10 + 16);
      }
      *(_QWORD *)v10 = a2;
      if ( a2 )
      {
        v9 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(v10 + 8) = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = v10 + 8;
        *(_QWORD *)(v10 + 16) = v14;
        *(_QWORD *)(a2 + 16) = v10;
      }
      ++v15;
    }
  }
  return v15;
}
