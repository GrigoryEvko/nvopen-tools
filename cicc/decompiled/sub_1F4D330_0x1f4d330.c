// Function: sub_1F4D330
// Address: 0x1f4d330
//
bool __fastcall sub_1F4D330(unsigned __int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  char v6; // r14
  __int64 v9; // rax
  bool result; // al
  __int64 v11; // rdx
  __int64 v12; // rax
  __int16 v13; // ax

  v6 = a5;
  while ( a2 <= 0 )
  {
    result = sub_1F4D060(a1, a2, a4, a4, a5, a6);
    if ( !result )
      return result;
    if ( a2 )
      v11 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
    else
      v11 = **(_QWORD **)(a3 + 272);
    if ( v11 )
    {
      if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 )
      {
        v11 = *(_QWORD *)(v11 + 32);
        if ( v11 )
        {
          if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 )
            BUG();
        }
      }
    }
    v12 = *(_QWORD *)(v11 + 32);
    if ( v12 && (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
      return 1;
    a1 = *(_QWORD *)(v11 + 16);
    v13 = **(_WORD **)(a1 + 16);
    if ( v13 == 15 )
    {
      a2 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 48LL);
    }
    else
    {
      if ( (v13 & 0xFFFD) != 8 )
        return 1;
      a2 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 88LL);
    }
  }
  if ( v6 )
    return 1;
  v9 = *(_QWORD *)(*(_QWORD *)(a3 + 272) + 8LL * (unsigned int)a2);
  if ( v9 )
  {
    if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
    {
      v9 = *(_QWORD *)(v9 + 32);
      if ( v9 )
      {
        while ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 )
        {
LABEL_8:
          v9 = *(_QWORD *)(v9 + 32);
          if ( !v9 )
            return 1;
        }
        return sub_1F4D060(a1, a2, a4, a4, a5, a6);
      }
      return 1;
    }
    while ( 1 )
    {
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        break;
      if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
        goto LABEL_8;
    }
  }
  return sub_1F4D060(a1, a2, a4, a4, a5, a6);
}
