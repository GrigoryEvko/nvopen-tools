// Function: sub_1795F90
// Address: 0x1795f90
//
__int64 __fastcall sub_1795F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  char v5; // al
  _BYTE *v8; // rdi
  unsigned __int8 v9; // al
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 52 )
  {
    if ( *(_QWORD *)a1 == *(_QWORD *)(a2 - 48) )
    {
      v8 = *(_BYTE **)(a2 - 24);
      v9 = v8[16];
      if ( v9 == 13 )
        goto LABEL_8;
      LOBYTE(v4) = v9 <= 0x10u && *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16;
      if ( (_BYTE)v4 )
      {
        v13 = sub_15A1020(v8, a2, *(_QWORD *)v8, a4);
        if ( v13 )
        {
          if ( *(_BYTE *)(v13 + 16) == 13 )
          {
            **(_QWORD **)(a1 + 8) = v13 + 24;
            return v4;
          }
        }
      }
    }
  }
  else if ( v5 == 5 && *(_WORD *)(a2 + 18) == 28 )
  {
    v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v11 = *(_QWORD *)(a2 - 24 * v10);
    if ( *(_QWORD *)a1 == v11 )
    {
      v8 = *(_BYTE **)(a2 + 24 * (1 - v10));
      if ( v8[16] != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 )
        {
          v12 = sub_15A1020(v8, a2, 1 - v10, v11);
          if ( v12 )
          {
            if ( *(_BYTE *)(v12 + 16) == 13 )
            {
              v4 = 1;
              **(_QWORD **)(a1 + 8) = v12 + 24;
              return v4;
            }
          }
        }
        return 0;
      }
LABEL_8:
      v4 = 1;
      **(_QWORD **)(a1 + 8) = v8 + 24;
      return v4;
    }
  }
  return 0;
}
