// Function: sub_3962650
// Address: 0x3962650
//
__int64 __fastcall sub_3962650(__int64 a1, float a2)
{
  _BOOL4 v2; // edx
  unsigned int v4; // edx
  _QWORD *v5; // rdx
  _QWORD *v6; // rcx
  _QWORD *v7; // rax
  unsigned __int8 v8; // si
  char v9; // r8
  int *v10; // rdx

  if ( *(_BYTE *)(a1 + 48) )
  {
    if ( !*(_DWORD *)(a1 + 128) )
      return 0;
    v5 = *(_QWORD **)(a1 + 120);
    v6 = &v5[2 * *(unsigned int *)(a1 + 136)];
    if ( v5 == v6 )
      return 0;
    while ( 1 )
    {
      v7 = v5;
      v8 = *v5 == -16 || *v5 == -8;
      if ( !v8 )
        break;
      v5 += 2;
      if ( v6 == v5 )
        return 0;
    }
    if ( v5 == v6 )
    {
      return 0;
    }
    else
    {
      v9 = 0;
      while ( 1 )
      {
        v10 = (int *)v7[1];
        v9 |= (float)v10[1] > (float)((float)v10[5] * a2);
        v8 |= (float)*v10 > (float)((float)v10[4] * a2);
        if ( v8 )
        {
          if ( v9 )
            return 257;
        }
        v7 += 2;
        if ( v7 != v6 )
        {
          while ( *v7 == -16 || *v7 == -8 )
          {
            v7 += 2;
            if ( v6 == v7 )
              goto LABEL_18;
          }
          if ( v7 != v6 )
            continue;
        }
LABEL_18:
        v4 = v8;
        BYTE1(v4) = v9;
        return v4;
      }
    }
  }
  else
  {
    v2 = (float)*(int *)(a1 + 24) > (float)(a2 * (float)*(int *)(a1 + 40));
    BYTE1(v2) = (float)*(int *)(a1 + 28) > (float)((float)*(int *)(a1 + 44) * a2);
    return v2;
  }
}
