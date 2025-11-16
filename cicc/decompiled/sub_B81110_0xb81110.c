// Function: sub_B81110
// Address: 0xb81110
//
__int64 __fastcall sub_B81110(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r9
  char v5; // r8
  unsigned int v6; // ecx
  __int64 v7; // r11
  unsigned int v9; // ebx
  __int64 v10; // r11

  v3 = *(unsigned int *)(a1 + 232);
  v4 = *(_QWORD *)(a1 + 216);
  v5 = a3;
  if ( (_DWORD)v3 )
  {
    v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    a3 = v4 + 16LL * v6;
    v7 = *(_QWORD *)a3;
    if ( a2 == *(_QWORD *)a3 )
    {
      if ( a3 != v4 + 16 * v3 )
        return *(_QWORD *)(a3 + 8);
    }
    else
    {
      a3 = 1;
      if ( v7 != -4096 )
      {
        while ( 1 )
        {
          v9 = a3 + 1;
          v6 = (v3 - 1) & (a3 + v6);
          a3 = v4 + 16LL * v6;
          v10 = *(_QWORD *)a3;
          if ( a2 == *(_QWORD *)a3 )
            break;
          a3 = v9;
          if ( v10 == -4096 )
            goto LABEL_13;
        }
        if ( a3 != v4 + 16 * v3 )
          return *(_QWORD *)(a3 + 8);
LABEL_13:
        if ( v5 )
          return sub_B811E0(*(_QWORD *)(a1 + 8), a2, a3);
        return 0;
      }
    }
  }
  if ( v5 )
    return sub_B811E0(*(_QWORD *)(a1 + 8), a2, a3);
  return 0;
}
