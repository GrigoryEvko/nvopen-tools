// Function: sub_10C8FF0
// Address: 0x10c8ff0
//
__int64 __fastcall sub_10C8FF0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  _BYTE *v6; // rax
  __int64 v7; // rsi
  __int64 v9; // rcx
  __int64 v10; // r12
  unsigned int v11; // eax
  __int64 v12; // rsi
  unsigned int v13; // r13d
  bool v14; // al
  __int64 *v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdx
  _BYTE *v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rdi
  unsigned int v21; // r13d
  int v22; // r13d
  char v23; // r14
  unsigned int v24; // r15d
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // r8
  unsigned int v28; // r14d

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 8) )
    return 0;
  if ( *(_BYTE *)v4 != 54 )
    return 0;
  v6 = *(_BYTE **)(v4 - 64);
  if ( *v6 != 68 )
    return 0;
  v7 = *((_QWORD *)v6 - 4);
  if ( !v7 )
    return 0;
  **a1 = v7;
  v9 = *(_QWORD *)(v4 - 32);
  if ( !v9 )
    return 0;
  *a1[1] = v9;
  v10 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    v12 = *(_QWORD *)(v10 + 24);
    v13 = v11 - 1;
    if ( v11 > 0x40 )
    {
      if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & (1LL << v13)) == 0 || v13 != (unsigned int)sub_C44590(v10 + 24) )
        return 0;
      goto LABEL_16;
    }
    v14 = v12 == 1LL << v13;
  }
  else
  {
    v16 = *(_QWORD *)(v10 + 8);
    v17 = (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17;
    if ( (unsigned int)v17 > 1 || *(_BYTE *)v10 > 0x15u )
      return 0;
    v18 = sub_AD7630(v10, 0, v17);
    if ( !v18 || *v18 != 17 )
    {
      if ( *(_BYTE *)(v16 + 8) == 17 )
      {
        v22 = *(_DWORD *)(v16 + 32);
        if ( v22 )
        {
          v23 = 0;
          v24 = 0;
          while ( 1 )
          {
            v25 = sub_AD69F0((unsigned __int8 *)v10, v24);
            if ( !v25 )
              break;
            if ( *(_BYTE *)v25 != 13 )
            {
              if ( *(_BYTE *)v25 != 17 )
                return 0;
              v26 = *(_DWORD *)(v25 + 32);
              v27 = *(_QWORD *)(v25 + 24);
              v28 = v26 - 1;
              if ( v26 <= 0x40 )
              {
                if ( v27 != 1LL << v28 )
                  return 0;
              }
              else if ( (*(_QWORD *)(v27 + 8LL * (v28 >> 6)) & (1LL << v28)) == 0
                     || (unsigned int)sub_C44590(v25 + 24) != v28 )
              {
                return 0;
              }
              v23 = 1;
            }
            if ( v22 == ++v24 )
            {
              if ( v23 )
                goto LABEL_16;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v19 = *((_DWORD *)v18 + 8);
    v20 = *((_QWORD *)v18 + 3);
    v21 = v19 - 1;
    if ( v19 > 0x40 )
    {
      if ( (*(_QWORD *)(v20 + 8LL * (v21 >> 6)) & (1LL << v21)) == 0
        || v21 != (unsigned int)sub_C44590((__int64)(v18 + 24)) )
      {
        return 0;
      }
      goto LABEL_16;
    }
    v14 = v20 == 1LL << v21;
  }
  if ( !v14 )
    return 0;
LABEL_16:
  v15 = a1[2];
  if ( v15 )
    *v15 = v10;
  return 1;
}
