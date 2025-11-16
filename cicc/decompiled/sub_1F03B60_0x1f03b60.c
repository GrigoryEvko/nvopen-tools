// Function: sub_1F03B60
// Address: 0x1f03b60
//
__int64 __fastcall sub_1F03B60(__int64 *a1, unsigned int a2)
{
  __int64 result; // rax
  unsigned int v4; // r9d
  unsigned int v5; // esi
  unsigned int v6; // r8d
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned int v9; // r11d
  __int64 v10; // rcx
  __int64 v11; // rax
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r10
  unsigned int *v16; // r9
  unsigned int *v17; // rax
  unsigned int v18; // r13d
  unsigned int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // r14

  result = a1[26];
  v4 = *((_DWORD *)a1 + 2);
  v5 = *(unsigned __int16 *)(result + 2LL * a2);
  if ( v5 < v4 )
  {
    v6 = a2;
    v7 = *a1;
    while ( 1 )
    {
      result = v5;
      v8 = v7 + 24LL * v5;
      if ( v6 == *(_DWORD *)(v8 + 12) )
      {
        v9 = *(_DWORD *)(v8 + 16);
        if ( v9 != -1 )
        {
          v10 = v9;
          if ( *(_DWORD *)(v7 + 24LL * v9 + 20) == -1 )
            break;
        }
      }
      v5 += 0x10000;
      if ( v4 <= v5 )
        return result;
    }
    if ( v5 != -1 )
    {
      while ( 1 )
      {
        v14 = 24 * result;
        v15 = 24 * v10;
        v16 = (unsigned int *)(v7 + 24 * result);
        v17 = (unsigned int *)(v7 + 24 * v10);
        if ( v16 == v17 )
          break;
        if ( v17[5] == -1 )
        {
          *(_WORD *)(a1[26] + 2LL * v16[3]) = v16[5];
          *(_DWORD *)(*a1 + 24LL * v16[5] + 16) = v16[4];
          v12 = v16[5];
          v13 = v14 + *a1;
        }
        else
        {
          v11 = v16[5];
          if ( (_DWORD)v11 == -1 )
          {
            v18 = *((_DWORD *)a1 + 2);
            v19 = *(unsigned __int16 *)(a1[26] + 2LL * v16[3]);
            if ( v19 < v18 )
            {
              while ( 1 )
              {
                v20 = v7 + 24LL * v19;
                if ( v16[3] == *(_DWORD *)(v20 + 12) )
                {
                  v21 = *(unsigned int *)(v20 + 16);
                  if ( (_DWORD)v21 != -1 && *(_DWORD *)(v7 + 24 * v21 + 20) == -1 )
                    break;
                }
                v19 += 0x10000;
                if ( v18 <= v19 )
                  goto LABEL_25;
              }
            }
            else
            {
LABEL_25:
              v20 = v7 + 0x17FFFFFFE8LL;
            }
            *(_DWORD *)(v20 + 16) = v9;
            *(_DWORD *)(*a1 + v15 + 20) = v16[5];
            v12 = *(_DWORD *)(*a1 + 24LL * v16[4] + 20);
            v13 = v14 + *a1;
          }
          else
          {
            *(_DWORD *)(v7 + 24 * v11 + 16) = v9;
            v12 = v16[5];
            *(_DWORD *)(*a1 + v15 + 20) = v12;
            v13 = v14 + *a1;
          }
        }
        *(_DWORD *)(v13 + 16) = -1;
        result = *a1;
        *(_DWORD *)(*a1 + v14 + 20) = *((_DWORD *)a1 + 56);
        *((_DWORD *)a1 + 56) = v5;
        ++*((_DWORD *)a1 + 57);
        if ( v12 == -1 )
          return result;
        result = v12;
        v7 = *a1;
        v9 = *(_DWORD *)(*a1 + 24LL * v12 + 16);
        v5 = v12;
        v10 = v9;
      }
      v16[4] = -1;
      result = *a1;
      *(_DWORD *)(*a1 + v14 + 20) = *((_DWORD *)a1 + 56);
      ++*((_DWORD *)a1 + 57);
      *((_DWORD *)a1 + 56) = v5;
    }
  }
  return result;
}
