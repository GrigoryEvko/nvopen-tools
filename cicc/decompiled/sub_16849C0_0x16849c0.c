// Function: sub_16849C0
// Address: 0x16849c0
//
__int64 __fastcall sub_16849C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v6; // ax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 i; // r12
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax

  v6 = *(_WORD *)(a1 + 84) >> 4;
  if ( (_BYTE)v6 == 1 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 104)
                   + 8LL
                   * (*(_DWORD *)(a1 + 40)
                    & ((unsigned int)(a2 >> 11)
                     ^ (unsigned int)(a2 >> 8)
                     ^ (unsigned int)(a2 >> 5))));
    if ( v8 )
    {
      while ( 1 )
      {
        v9 = *(unsigned int *)(v8 + 4);
        v8 += 4;
        if ( (_DWORD)v9 == -1 )
          break;
        if ( a2 == *(_QWORD *)(*(_QWORD *)(a1 + 88) + 16 * v9) )
          return 1;
      }
    }
  }
  else if ( (_BYTE)v6 == 2 )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & (unsigned int)a2));
    if ( v17 )
    {
      while ( 1 )
      {
        v18 = *(unsigned int *)(v17 + 4);
        v17 += 4;
        if ( (_DWORD)v18 == -1 )
          break;
        if ( a2 == *(_QWORD *)(*(_QWORD *)(a1 + 88) + 16 * v18) )
          return 1;
      }
    }
  }
  else
  {
    if ( (_BYTE)v6 )
      return 0;
    v10 = *(_QWORD *)(a1 + 32);
    if ( v10 )
      v11 = (*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, __int64, _QWORD))(a1 + 16))(
              a2,
              v10,
              a3,
              a4,
              0);
    else
      v11 = (*(__int64 (__fastcall **)(unsigned __int64, _QWORD, __int64, __int64, _QWORD))a1)(a2, 0, a3, a4, 0);
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & v11));
    if ( v12 )
    {
      v13 = v12 + 4;
      v14 = *(unsigned int *)(v12 + 4);
      for ( i = *(_QWORD *)(a1 + 88); (_DWORD)v14 != -1; v13 += 4 )
      {
        v16 = *(_QWORD *)(i + 16 * v14);
        if ( *(_QWORD *)(a1 + 32) )
        {
          if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 24))(v16, a2) )
            return 1;
        }
        else if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 8))(v16, a2) )
        {
          return 1;
        }
        v14 = *(unsigned int *)(v13 + 4);
      }
    }
  }
  return 0;
}
