// Function: sub_1D85D90
// Address: 0x1d85d90
//
_BOOL8 __fastcall sub_1D85D90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 (*v3)(void); // rax
  __int64 v4; // r15
  _BOOL8 v5; // r8
  __int64 v6; // rsi
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r13
  _BOOL4 v11; // r12d
  __int64 v13; // rax
  __int64 i; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 56LL);
  if ( v3 != sub_1D12D20 )
    v2 = v3();
  v4 = *(_QWORD *)(a2 + 328);
  v5 = 0;
  for ( i = a2 + 320; i != v4; v4 = *(_QWORD *)(v8 + 8) )
  {
    v6 = *(_QWORD *)(v4 + 32);
    v7 = v4 + 24;
    v8 = v4;
    while ( v6 != v7 )
    {
      while ( 1 )
      {
        if ( !v6 )
          BUG();
        v9 = v6;
        if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 46) & 8) != 0 )
        {
          do
            v9 = *(_QWORD *)(v9 + 8);
          while ( (*(_BYTE *)(v9 + 46) & 8) != 0 );
        }
        v10 = *(_QWORD *)(v9 + 8);
        v11 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL) & 0x800000LL) != 0;
        if ( (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL) & 0x800000LL) != 0 )
          break;
        v6 = *(_QWORD *)(v9 + 8);
        if ( v10 == v7 )
          goto LABEL_12;
      }
      v15 = v8;
      v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _BOOL8))(*(_QWORD *)v2 + 1456LL))(
              v2,
              v6,
              v4,
              v8,
              v5);
      v8 = v15;
      v6 = v10;
      v5 = v11;
      if ( v4 != v13 )
      {
        v6 = *(_QWORD *)(v13 + 32);
        v8 = v13;
        v7 = v13 + 24;
        v4 = v13;
      }
    }
LABEL_12:
    ;
  }
  return v5;
}
