// Function: sub_32A0AD0
// Address: 0x32a0ad0
//
bool __fastcall sub_32A0AD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  int v8; // r9d
  __int64 v9; // r8
  int v10; // r11d
  __int64 v11; // r10

  result = 0;
  if ( *(_DWORD *)a4 == *(_DWORD *)(a1 + 24) )
  {
    v5 = *(__int64 **)(a1 + 40);
    v6 = *(_QWORD *)(a4 + 8);
    v7 = *v5;
    v8 = *((_DWORD *)v5 + 2);
    v9 = v5[5];
    if ( v6 )
    {
      if ( v7 != v6 || v8 != *(_DWORD *)(a4 + 16) )
      {
        v10 = *((_DWORD *)v5 + 12);
        goto LABEL_6;
      }
    }
    else if ( !v7 )
    {
      goto LABEL_19;
    }
    v11 = *(_QWORD *)(a4 + 24);
    if ( !v11 )
    {
      if ( v9 )
        goto LABEL_11;
      return 0;
    }
    v10 = *((_DWORD *)v5 + 12);
    if ( v11 == v9 )
    {
      if ( v10 == *(_DWORD *)(a4 + 32) )
        goto LABEL_11;
      if ( !v6 )
        goto LABEL_9;
      goto LABEL_6;
    }
    if ( v6 )
    {
LABEL_6:
      result = 0;
      if ( v6 != v9 || *(_DWORD *)(a4 + 16) != v10 )
        return result;
LABEL_8:
      v11 = *(_QWORD *)(a4 + 24);
      if ( !v11 )
      {
        result = 0;
        if ( !v7 )
          return result;
        goto LABEL_11;
      }
LABEL_9:
      result = 0;
      if ( v7 != v11 || v8 != *(_DWORD *)(a4 + 32) )
        return result;
LABEL_11:
      result = 1;
      if ( *(_BYTE *)(a4 + 44) )
        return (*(_DWORD *)(a4 + 40) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 40);
      return result;
    }
LABEL_19:
    if ( v9 )
      goto LABEL_8;
    return 0;
  }
  return result;
}
