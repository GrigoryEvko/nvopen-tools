// Function: sub_10C4470
// Address: 0x10c4470
//
__int64 __fastcall sub_10C4470(__int64 **a1, __int64 a2)
{
  unsigned int v4; // eax
  __int64 v5; // rsi
  unsigned int v6; // r13d
  __int64 v8; // r13
  __int64 v9; // rdx
  _BYTE *v10; // rax
  unsigned int v11; // edx
  __int64 v12; // rdi
  unsigned int v13; // r13d
  bool v14; // al
  int v15; // r13d
  char v16; // r14
  unsigned int v17; // r15d
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // r8
  unsigned int v21; // r14d

  if ( *(_BYTE *)a2 == 17 )
  {
    v4 = *(_DWORD *)(a2 + 32);
    v5 = *(_QWORD *)(a2 + 24);
    v6 = v4 - 1;
    if ( v4 > 0x40 )
    {
      if ( (*(_QWORD *)(v5 + 8LL * (v6 >> 6)) & (1LL << v6)) == 0 && (unsigned int)sub_C445E0(a2 + 24) == v6 )
        goto LABEL_15;
      return 0;
    }
    v14 = v5 == (1LL << v6) - 1;
LABEL_14:
    if ( v14 )
      goto LABEL_15;
    return 0;
  }
  v8 = *(_QWORD *)(a2 + 8);
  v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
  if ( (unsigned int)v9 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v10 = sub_AD7630(a2, 0, v9);
  if ( !v10 || *v10 != 17 )
  {
    if ( *(_BYTE *)(v8 + 8) == 17 )
    {
      v15 = *(_DWORD *)(v8 + 32);
      if ( v15 )
      {
        v16 = 0;
        v17 = 0;
        while ( 1 )
        {
          v18 = sub_AD69F0((unsigned __int8 *)a2, v17);
          if ( !v18 )
            break;
          if ( *(_BYTE *)v18 != 13 )
          {
            if ( *(_BYTE *)v18 != 17 )
              return 0;
            v19 = *(_DWORD *)(v18 + 32);
            v20 = *(_QWORD *)(v18 + 24);
            v21 = v19 - 1;
            if ( v19 <= 0x40 )
            {
              if ( v20 != (1LL << v21) - 1 )
                return 0;
            }
            else if ( (*(_QWORD *)(v20 + 8LL * (v21 >> 6)) & (1LL << v21)) != 0
                   || (unsigned int)sub_C445E0(v18 + 24) != v21 )
            {
              return 0;
            }
            v16 = 1;
          }
          if ( v15 == ++v17 )
          {
            if ( v16 )
              goto LABEL_15;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v11 = *((_DWORD *)v10 + 8);
  v12 = *((_QWORD *)v10 + 3);
  v13 = v11 - 1;
  if ( v11 <= 0x40 )
  {
    v14 = v12 == (1LL << v13) - 1;
    goto LABEL_14;
  }
  if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & (1LL << v13)) != 0 || v13 != (unsigned int)sub_C445E0((__int64)(v10 + 24)) )
    return 0;
LABEL_15:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
