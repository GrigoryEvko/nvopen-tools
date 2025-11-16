// Function: sub_1105FC0
// Address: 0x1105fc0
//
__int64 __fastcall sub_1105FC0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rdi
  int v9; // r10d
  int v10; // r9d
  __int64 v11; // r12
  unsigned int v12; // eax
  __int64 v13; // rsi
  unsigned int v14; // r13d
  __int64 *v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdx
  _BYTE *v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rdi
  unsigned int v21; // r13d
  bool v22; // al
  int v23; // r13d
  char v24; // r14
  unsigned int v25; // r15d
  __int64 v26; // rax
  unsigned int v27; // edx
  __int64 v28; // r8
  unsigned int v29; // r14d

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v4 != 78 )
    return 0;
  v6 = *(_QWORD *)(v4 - 32);
  v7 = *(_QWORD *)(v4 + 8);
  v8 = *(_QWORD *)(v6 + 8);
  v9 = *(unsigned __int8 *)(v7 + 8);
  v10 = *(unsigned __int8 *)(v8 + 8);
  if ( (unsigned int)(v10 - 17) <= 1 != (unsigned int)(v9 - 17) <= 1
    || (unsigned int)(v10 - 17) <= 1
    && (*(_DWORD *)(v7 + 32) != *(_DWORD *)(v8 + 32) || ((_BYTE)v9 == 18) != ((_BYTE)v10 == 18)) )
  {
    return 0;
  }
  **a1 = v6;
  v11 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v11 == 17 )
  {
    v12 = *(_DWORD *)(v11 + 32);
    v13 = *(_QWORD *)(v11 + 24);
    v14 = v12 - 1;
    if ( v12 > 0x40 )
    {
      if ( (*(_QWORD *)(v13 + 8LL * (v14 >> 6)) & (1LL << v14)) != 0 && (unsigned int)sub_C44590(v11 + 24) == v14 )
        goto LABEL_12;
      return 0;
    }
    v22 = v13 == 1LL << v14;
LABEL_24:
    if ( v22 )
      goto LABEL_12;
    return 0;
  }
  v16 = *(_QWORD *)(v11 + 8);
  v17 = (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17;
  if ( (unsigned int)v17 > 1 || *(_BYTE *)v11 > 0x15u )
    return 0;
  v18 = sub_AD7630(v11, 0, v17);
  if ( !v18 || *v18 != 17 )
  {
    if ( *(_BYTE *)(v16 + 8) == 17 )
    {
      v23 = *(_DWORD *)(v16 + 32);
      if ( v23 )
      {
        v24 = 0;
        v25 = 0;
        while ( 1 )
        {
          v26 = sub_AD69F0((unsigned __int8 *)v11, v25);
          if ( !v26 )
            break;
          if ( *(_BYTE *)v26 != 13 )
          {
            if ( *(_BYTE *)v26 != 17 )
              return 0;
            v27 = *(_DWORD *)(v26 + 32);
            v28 = *(_QWORD *)(v26 + 24);
            v29 = v27 - 1;
            if ( v27 <= 0x40 )
            {
              if ( v28 != 1LL << v29 )
                return 0;
            }
            else if ( (*(_QWORD *)(v28 + 8LL * (v29 >> 6)) & (1LL << v29)) == 0
                   || (unsigned int)sub_C44590(v26 + 24) != v29 )
            {
              return 0;
            }
            v24 = 1;
          }
          if ( v23 == ++v25 )
          {
            if ( v24 )
              goto LABEL_12;
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
  if ( v19 <= 0x40 )
  {
    v22 = v20 == 1LL << v21;
    goto LABEL_24;
  }
  if ( (*(_QWORD *)(v20 + 8LL * (v21 >> 6)) & (1LL << v21)) == 0 || (unsigned int)sub_C44590((__int64)(v18 + 24)) != v21 )
    return 0;
LABEL_12:
  v15 = a1[1];
  if ( v15 )
    *v15 = v11;
  return 1;
}
