// Function: sub_98C610
// Address: 0x98c610
//
__int64 __fastcall sub_98C610(char *a1, __int64 a2, __int64 a3)
{
  char v3; // cl
  char v4; // cl
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // edx

  v3 = *a1;
  if ( (unsigned __int8)*a1 > 0x1Cu )
  {
    if ( v3 == 85 )
    {
      v6 = *((_QWORD *)a1 - 4);
      v4 = 51;
      if ( v6 )
      {
        if ( !*(_BYTE *)v6 )
        {
          v7 = *((_QWORD *)a1 + 10);
          if ( *(_QWORD *)(v6 + 24) == v7 && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
            return sub_9B7470(*(unsigned int *)(v6 + 36), v7, a3, 51);
        }
      }
      return ((0x108100000000041uLL >> v4) & 1) == 0;
    }
    if ( v3 == 92 )
    {
      v8 = *((_DWORD *)a1 + 20);
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)a1 - 8) + 8LL) + 32LL) != v8
        || !(unsigned __int8)sub_B4EEA0(*((_QWORD *)a1 + 9), v8) )
      {
        return 0;
      }
      v3 = *a1;
    }
  }
  v4 = v3 - 34;
  result = 1;
  if ( (unsigned __int8)v4 <= 0x38u )
    return ((0x108100000000041uLL >> v4) & 1) == 0;
  return result;
}
