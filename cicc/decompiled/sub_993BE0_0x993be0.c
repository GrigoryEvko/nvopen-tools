// Function: sub_993BE0
// Address: 0x993be0
//
__int64 __fastcall sub_993BE0(_QWORD **a1, __int64 a2)
{
  unsigned int v4; // eax
  __int64 v5; // rsi
  unsigned int v6; // r13d
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 v11; // rdi
  unsigned int v12; // r13d
  bool v13; // al
  int v14; // r13d
  char v15; // r14
  unsigned int v16; // r15d
  __int64 v17; // rax
  unsigned int v18; // edx
  __int64 v19; // r8
  unsigned int v20; // r14d

  if ( *(_BYTE *)a2 == 17 )
  {
    v4 = *(_DWORD *)(a2 + 32);
    v5 = *(_QWORD *)(a2 + 24);
    v6 = v4 - 1;
    if ( v4 > 0x40 )
    {
      if ( (*(_QWORD *)(v5 + 8LL * (v6 >> 6)) & (1LL << v6)) != 0 && (unsigned int)sub_C44590(a2 + 24) == v6 )
        goto LABEL_15;
      return 0;
    }
    v13 = v5 == 1LL << v6;
LABEL_14:
    if ( v13 )
      goto LABEL_15;
    return 0;
  }
  v8 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v9 = sub_AD7630(a2, 0);
  if ( !v9 || *(_BYTE *)v9 != 17 )
  {
    if ( *(_BYTE *)(v8 + 8) == 17 )
    {
      v14 = *(_DWORD *)(v8 + 32);
      if ( v14 )
      {
        v15 = 0;
        v16 = 0;
        while ( 1 )
        {
          v17 = sub_AD69F0(a2, v16);
          if ( !v17 )
            break;
          if ( *(_BYTE *)v17 != 13 )
          {
            if ( *(_BYTE *)v17 != 17 )
              return 0;
            v18 = *(_DWORD *)(v17 + 32);
            v19 = *(_QWORD *)(v17 + 24);
            v20 = v18 - 1;
            if ( v18 <= 0x40 )
            {
              if ( v19 != 1LL << v20 )
                return 0;
            }
            else if ( (*(_QWORD *)(v19 + 8LL * (v20 >> 6)) & (1LL << v20)) == 0
                   || (unsigned int)sub_C44590(v17 + 24) != v20 )
            {
              return 0;
            }
            v15 = 1;
          }
          if ( v14 == ++v16 )
          {
            if ( v15 )
              goto LABEL_15;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v10 = *(_DWORD *)(v9 + 32);
  v11 = *(_QWORD *)(v9 + 24);
  v12 = v10 - 1;
  if ( v10 <= 0x40 )
  {
    v13 = v11 == 1LL << v12;
    goto LABEL_14;
  }
  if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & (1LL << v12)) == 0 || v12 != (unsigned int)sub_C44590(v9 + 24) )
    return 0;
LABEL_15:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
