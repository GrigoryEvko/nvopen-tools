// Function: sub_2FF8D40
// Address: 0x2ff8d40
//
__int64 __fastcall sub_2FF8D40(int a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r9
  int v4; // r10d
  int v5; // esi
  int *v6; // r8
  unsigned int v7; // edx
  int *v8; // rax
  int v9; // ecx
  int v10; // eax
  __int64 result; // rax
  int v12; // r11d

  if ( a1 >= 0 )
  {
LABEL_11:
    result = 0;
    if ( (unsigned int)(a1 - 1) <= 0x3FFFFFFE )
      return (unsigned int)a1;
  }
  else
  {
    v2 = *(unsigned int *)(a2 + 24);
    v3 = *(_QWORD *)(a2 + 8);
    v4 = v2 - 1;
    v5 = *(_DWORD *)(a2 + 24);
    v6 = (int *)(v3 + 8 * v2);
    while ( v5 )
    {
      v7 = v4 & (37 * a1);
      v8 = (int *)(v3 + 8LL * v7);
      v9 = *v8;
      if ( a1 != *v8 )
      {
        v10 = 1;
        while ( v9 != -1 )
        {
          v12 = v10 + 1;
          v7 = v4 & (v10 + v7);
          v8 = (int *)(v3 + 8LL * v7);
          v9 = *v8;
          if ( a1 == *v8 )
            goto LABEL_4;
          v10 = v12;
        }
        return 0;
      }
LABEL_4:
      if ( v6 == v8 )
        return 0;
      a1 = v8[1];
      if ( a1 >= 0 )
        goto LABEL_11;
    }
    return 0;
  }
  return result;
}
