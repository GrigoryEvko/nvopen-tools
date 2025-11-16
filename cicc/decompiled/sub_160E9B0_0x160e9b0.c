// Function: sub_160E9B0
// Address: 0x160e9b0
//
__int64 __fastcall sub_160E9B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  char v4; // r8
  __int64 v5; // r9
  unsigned int v6; // ecx
  __int64 v7; // r11
  unsigned int v9; // ebx
  __int64 v10; // r11

  v3 = *(unsigned int *)(a1 + 248);
  v4 = a3;
  if ( (_DWORD)v3 )
  {
    v5 = *(_QWORD *)(a1 + 232);
    v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    a3 = v5 + 16LL * v6;
    v7 = *(_QWORD *)a3;
    if ( a2 == *(_QWORD *)a3 )
    {
      if ( a3 != v5 + 16 * v3 )
        return *(_QWORD *)(a3 + 8);
    }
    else
    {
      a3 = 1;
      if ( v7 != -4 )
      {
        while ( 1 )
        {
          v9 = a3 + 1;
          v6 = (v3 - 1) & (a3 + v6);
          a3 = v5 + 16LL * v6;
          v10 = *(_QWORD *)a3;
          if ( a2 == *(_QWORD *)a3 )
            break;
          a3 = v9;
          if ( v10 == -4 )
            goto LABEL_13;
        }
        if ( a3 != v5 + 16 * v3 )
          return *(_QWORD *)(a3 + 8);
LABEL_13:
        if ( v4 )
          return sub_160EA80(*(_QWORD *)(a1 + 16), a2, a3);
        return 0;
      }
    }
  }
  if ( v4 )
    return sub_160EA80(*(_QWORD *)(a1 + 16), a2, a3);
  return 0;
}
