// Function: sub_1684840
// Address: 0x1684840
//
__int64 __fastcall sub_1684840(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v6; // ax
  __int64 v7; // r8
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 i; // r15
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rbx

  v6 = *(_WORD *)(a1 + 84) >> 4;
  if ( (_BYTE)v6 == 1 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 104)
                   + 8LL
                   * (*(_DWORD *)(a1 + 40)
                    & ((unsigned int)(a2 >> 11)
                     ^ (unsigned int)(a2 >> 8)
                     ^ (unsigned int)(a2 >> 5))));
    if ( v9 )
    {
      while ( 1 )
      {
        v11 = *(unsigned int *)(v9 + 4);
        v9 += 4;
        if ( (_DWORD)v11 == -1 )
          break;
        v10 = (__int64 *)(*(_QWORD *)(a1 + 88) + 16 * v11);
        if ( a2 == *v10 )
          return v10[1];
      }
    }
    return 0;
  }
  if ( (_BYTE)v6 != 2 )
  {
    v7 = 0;
    if ( (_BYTE)v6 )
      return v7;
    v12 = *(_QWORD *)(a1 + 32);
    if ( v12 )
      v13 = (*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, __int64, _QWORD))(a1 + 16))(
              a2,
              v12,
              a3,
              a4,
              0);
    else
      v13 = (*(__int64 (__fastcall **)(unsigned __int64, _QWORD, __int64, __int64, _QWORD))a1)(a2, 0, a3, a4, 0);
    v14 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & v13));
    if ( v14 )
    {
      v15 = *(unsigned int *)(v14 + 4);
      v16 = *(_QWORD *)(a1 + 88);
      for ( i = v14 + 4; (_DWORD)v15 != -1; i += 4 )
      {
        v10 = (__int64 *)(v16 + 16 * v15);
        v18 = *v10;
        if ( *(_QWORD *)(a1 + 32) )
        {
          if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 24))(v18, a2) )
            return v10[1];
        }
        else if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 8))(v18, a2) )
        {
          return v10[1];
        }
        v15 = *(unsigned int *)(i + 4);
      }
    }
    return 0;
  }
  v19 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & (unsigned int)a2));
  if ( !v19 )
    return 0;
  while ( 1 )
  {
    v20 = *(unsigned int *)(v19 + 4);
    v19 += 4;
    if ( (_DWORD)v20 == -1 )
      break;
    v10 = (__int64 *)(*(_QWORD *)(a1 + 88) + 16 * v20);
    if ( a2 == *v10 )
      return v10[1];
  }
  return 0;
}
