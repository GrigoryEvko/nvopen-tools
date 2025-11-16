// Function: sub_F57430
// Address: 0xf57430
//
__int64 __fastcall sub_F57430(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 (__fastcall *a5)(__int64, __int64, __int64),
        __int64 a6)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v17; // [rsp+1Ch] [rbp-34h]

  v8 = *(_QWORD *)(a1 + 16);
  v17 = 0;
  while ( v8 )
  {
    while ( 1 )
    {
      v11 = v8;
      v8 = *(_QWORD *)(v8 + 8);
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
      if ( !v8 )
        return v17;
    }
    if ( (unsigned __int8)sub_B19ED0(a3, a4, v11) && a5(a6, v11, a2) )
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
        *(_QWORD *)(v11 + 16) = a2 + 16;
        *(_QWORD *)(a2 + 16) = v11;
      }
      ++v17;
    }
  }
  return v17;
}
