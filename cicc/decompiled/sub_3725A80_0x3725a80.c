// Function: sub_3725A80
// Address: 0x3725a80
//
__int64 __fastcall sub_3725A80(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 i; // rdx
  __int64 v11; // rbx
  __int64 v12; // r15
  int v13; // eax
  int v14; // edi
  __int64 *v15; // r10
  unsigned int j; // edx
  __int64 *v17; // rax
  __int64 v18; // r11
  __int64 v19; // rdx
  unsigned int v20; // edx
  __int64 v21; // rdx
  __int64 k; // rdx
  __int64 v23; // [rsp+8h] [rbp-58h]
  int v24; // [rsp+1Ch] [rbp-44h]
  int v25; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v23 = 24 * v4;
    v9 = v5 + 24 * v4;
    for ( i = result + 24 * v8; i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = -1;
        *(_BYTE *)(result + 12) = 0;
      }
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      do
      {
        while ( *(_QWORD *)v11 == -1 )
        {
          if ( *(_DWORD *)(v11 + 8) != -1 )
            goto LABEL_13;
LABEL_24:
          if ( *(_BYTE *)(v11 + 12) )
            goto LABEL_13;
          v11 += 24;
          if ( v9 == v11 )
            return sub_C7D6A0(v5, v23, 8);
        }
        if ( *(_QWORD *)v11 == -2 && *(_DWORD *)(v11 + 8) == -2 )
          goto LABEL_24;
LABEL_13:
        v24 = *(_DWORD *)(a1 + 24);
        if ( !v24 )
        {
          qmemcpy(0, (const void *)v11, 0xDu);
          BUG();
        }
        v12 = *(_QWORD *)(a1 + 8);
        v25 = *(_DWORD *)(v11 + 8);
        v26[0] = *(_QWORD *)v11;
        v13 = sub_3723A60(v26, &v25, (_BYTE *)(v11 + 12));
        v14 = 1;
        v15 = 0;
        for ( j = (v24 - 1) & v13; ; j = (v24 - 1) & v20 )
        {
          v17 = (__int64 *)(v12 + 24LL * j);
          v18 = *v17;
          if ( *(_QWORD *)v11 == *v17
            && *(_DWORD *)(v11 + 8) == *((_DWORD *)v17 + 2)
            && *(_BYTE *)(v11 + 12) == *((_BYTE *)v17 + 12) )
          {
            break;
          }
          if ( v18 == -1 )
          {
            if ( *((_DWORD *)v17 + 2) == -1 && !*((_BYTE *)v17 + 12) )
            {
              if ( v15 )
                v17 = v15;
              break;
            }
          }
          else if ( v18 == -2 && *((_DWORD *)v17 + 2) == -2 && *((_BYTE *)v17 + 12) != 1 && !v15 )
          {
            v15 = (__int64 *)(v12 + 24LL * j);
          }
          v20 = v14 + j;
          ++v14;
        }
        v19 = *(_QWORD *)v11;
        v11 += 24;
        *v17 = v19;
        *((_DWORD *)v17 + 2) = *(_DWORD *)(v11 - 16);
        *((_BYTE *)v17 + 12) = *(_BYTE *)(v11 - 12);
        v17[2] = *(_QWORD *)(v11 - 8);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return sub_C7D6A0(v5, v23, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v21; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = -1;
        *(_BYTE *)(result + 12) = 0;
      }
    }
  }
  return result;
}
