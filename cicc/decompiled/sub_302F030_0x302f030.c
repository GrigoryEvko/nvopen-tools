// Function: sub_302F030
// Address: 0x302f030
//
__int64 __fastcall sub_302F030(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  _QWORD *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rdi
  signed __int64 v8; // rdx
  _QWORD *v9; // rcx

  v4 = *(_QWORD *)(a1 + 40);
  v5 = (_QWORD *)(v4 + 40 * a2);
  v6 = 5 * (*(unsigned int *)(a1 + 64) - (a3 + a2));
  v7 = &v5[v6];
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((v6 * 8) >> 3);
  if ( v8 >> 2 <= 0 )
  {
LABEL_11:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
          return *(_QWORD *)v4;
        goto LABEL_19;
      }
      if ( *(_DWORD *)(*v5 + 24LL) != 51 )
        goto LABEL_8;
      v5 += 5;
    }
    if ( *(_DWORD *)(*v5 + 24LL) != 51 )
      goto LABEL_8;
    v5 += 5;
LABEL_19:
    if ( *(_DWORD *)(*v5 + 24LL) == 51 )
      return *(_QWORD *)v4;
    goto LABEL_8;
  }
  v9 = &v5[20 * (v8 >> 2)];
  while ( *(_DWORD *)(*v5 + 24LL) == 51 )
  {
    if ( *(_DWORD *)(v5[5] + 24LL) != 51 )
    {
      v5 += 5;
      break;
    }
    if ( *(_DWORD *)(v5[10] + 24LL) != 51 )
    {
      v5 += 10;
      break;
    }
    if ( *(_DWORD *)(v5[15] + 24LL) != 51 )
    {
      v5 += 15;
      break;
    }
    v5 += 20;
    if ( v9 == v5 )
    {
      v8 = 0xCCCCCCCCCCCCCCCDLL * (v7 - v5);
      goto LABEL_11;
    }
  }
LABEL_8:
  if ( v7 != v5 )
    return 0;
  return *(_QWORD *)v4;
}
