// Function: sub_F57550
// Address: 0xf57550
//
__int64 __fastcall sub_F57550(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(__int64, __int64, __int64),
        __int64 a6)
{
  __int64 v8; // r15
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v18; // [rsp+1Ch] [rbp-34h]

  v8 = *(_QWORD *)(a1 + 16);
  v18 = 0;
  while ( v8 )
  {
    while ( 1 )
    {
      v12 = v8;
      v8 = *(_QWORD *)(v8 + 8);
      v13 = *(_QWORD *)(v12 + 24);
      if ( *(_BYTE *)v13 != 85 )
        break;
      v14 = *(_QWORD *)(v13 - 32);
      if ( !v14
        || *(_BYTE *)v14
        || *(_QWORD *)(v14 + 24) != *(_QWORD *)(v13 + 80)
        || (*(_BYTE *)(v14 + 33) & 0x20) == 0
        || *(_DWORD *)(v14 + 36) != 171 )
      {
        break;
      }
      if ( !v8 )
        return v18;
    }
    sub_B19BE0(a3, a4, v12);
    if ( v9 && a5(a6, v12, a2) )
    {
      if ( *(_QWORD *)v12 )
      {
        v10 = *(_QWORD *)(v12 + 8);
        **(_QWORD **)(v12 + 16) = v10;
        if ( v10 )
          *(_QWORD *)(v10 + 16) = *(_QWORD *)(v12 + 16);
      }
      *(_QWORD *)v12 = a2;
      if ( a2 )
      {
        v11 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(v12 + 8) = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = v12 + 8;
        *(_QWORD *)(v12 + 16) = a2 + 16;
        *(_QWORD *)(a2 + 16) = v12;
      }
      ++v18;
    }
  }
  return v18;
}
