// Function: sub_B4FA40
// Address: 0xb4fa40
//
__int64 __fastcall sub_B4FA40(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, __int64 a5)
{
  int v5; // r14d
  int v7; // eax
  int v9; // r13d
  unsigned __int64 v10; // rdx
  _DWORD *v13; // rdx
  __int64 i; // rcx
  __int64 j; // r11
  unsigned int v16; // esi
  int v17; // r10d
  int v18; // r9d
  int v19; // edx
  int v20; // edi
  int v21; // esi
  __int64 v22; // rcx
  unsigned int v23; // edx
  int v24; // edx
  int v25; // [rsp+Ch] [rbp-44h]
  __int64 v26; // [rsp+10h] [rbp-40h]

  v7 = a2 / a3;
  if ( a2 % a3 )
    return 0;
  if ( a2 < a3 )
    return 0;
  v9 = v7 - 1;
  if ( ((v7 - 1) & v7) != 0 )
    return 0;
  v10 = *(unsigned int *)(a5 + 8);
  if ( a3 != v10 )
  {
    if ( a3 >= v10 )
    {
      if ( a3 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        v25 = v7;
        v26 = a5;
        sub_C8D5F0(a5, a5 + 16, a3, 4);
        a5 = v26;
        v7 = v25;
        v10 = *(unsigned int *)(v26 + 8);
      }
      v13 = (_DWORD *)(*(_QWORD *)a5 + 4 * v10);
      for ( i = *(_QWORD *)a5 + 4LL * a3; (_DWORD *)i != v13; ++v13 )
      {
        if ( v13 )
          *v13 = 0;
      }
    }
    *(_DWORD *)(a5 + 8) = a3;
  }
  if ( a3 )
  {
    for ( j = 0; a3 != j; ++j )
    {
      if ( v9 )
      {
        v16 = j;
        v17 = 0;
        v18 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v19 = *(_DWORD *)(a1 + 4LL * v16);
            v20 = *(_DWORD *)(a1 + 4LL * (v16 + a3));
            v16 += a3;
            if ( v20 < 0 )
              break;
            if ( v19 < 0 )
              goto LABEL_32;
            if ( v19 + 1 != v20 )
              return 0;
LABEL_19:
            if ( ++v18 == v9 )
              goto LABEL_25;
          }
          if ( v19 < 0 )
          {
LABEL_32:
            if ( v17 )
            {
              ++v17;
              if ( v20 >= 0 && v17 + v5 != v20 )
                return 0;
            }
            goto LABEL_19;
          }
          ++v18;
          v5 = v19;
          v17 = 1;
          if ( v18 == v9 )
          {
LABEL_25:
            v21 = v9;
            goto LABEL_26;
          }
        }
      }
      v17 = 0;
      v21 = 0;
LABEL_26:
      v22 = 4 * j;
      if ( *(int *)(a1 + 4 * j) < 0 )
      {
        v24 = *(_DWORD *)(a1 + 4LL * (v9 * a3 + (unsigned int)j));
        if ( v24 < 0 )
        {
          if ( !v17 )
          {
            v23 = v7;
            goto LABEL_29;
          }
          v17 += v5 + 1 - v7;
        }
        else
        {
          v17 = v24 - v21;
        }
        if ( v17 < 0 )
          return 0;
      }
      else
      {
        v17 = *(_DWORD *)(a1 + 4 * j);
      }
      v23 = v7 + v17;
LABEL_29:
      if ( a4 < v23 )
        return 0;
      *(_DWORD *)(*(_QWORD *)a5 + v22) = v17;
    }
  }
  return 1;
}
