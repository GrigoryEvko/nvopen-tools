// Function: sub_3422830
// Address: 0x3422830
//
__int64 __fastcall sub_3422830(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6)
{
  __int64 v6; // rsi
  int v9; // eax
  char v10; // cl
  __int64 v12; // rdi
  __int64 v13; // rdx

  if ( !a5 )
    return 0;
  v6 = a4;
  v9 = *(_DWORD *)(a4 + 68);
  v10 = a6;
  if ( *(_WORD *)(*(_QWORD *)(v6 + 48) + 16LL * (unsigned int)(v9 - 1)) == 262 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v6 + 56);
      if ( !v13 )
        break;
      while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v13 + 48LL) + 16LL * *(unsigned int *)(v13 + 8)) != 262 )
      {
        v13 = *(_QWORD *)(v13 + 32);
        if ( !v13 )
          return (unsigned int)sub_341FE80(v6, a1, a3, v10) ^ 1;
      }
      v12 = *(_QWORD *)(v13 + 16);
      if ( !v12 )
        break;
      v10 = 0;
      if ( *(_WORD *)(*(_QWORD *)(v12 + 48) + 16LL * (unsigned int)(*(_DWORD *)(v12 + 68) - 1)) != 262 )
        return (unsigned int)sub_341FE80(v12, a1, a3, 0) ^ 1;
      v6 = *(_QWORD *)(v13 + 16);
    }
  }
  else
  {
    v10 = a6;
  }
  return (unsigned int)sub_341FE80(v6, a1, a3, v10) ^ 1;
}
