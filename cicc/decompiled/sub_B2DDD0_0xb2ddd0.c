// Function: sub_B2DDD0
// Address: 0xb2ddd0
//
__int64 __fastcall sub_B2DDD0(__int64 a1, _QWORD *a2, char a3, char a4, char a5, char a6, char a7)
{
  __int64 v7; // r13
  unsigned __int8 *v9; // r14
  int v10; // eax
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rcx
  unsigned int v21; // r15d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r8
  unsigned int v31; // ecx
  __int64 v32; // rdx
  __int64 v36; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v37; // [rsp+28h] [rbp-48h]
  int v38; // [rsp+30h] [rbp-40h]
  _BYTE v39[56]; // [rsp+38h] [rbp-38h] BYREF

  v7 = *(_QWORD *)(a1 + 16);
  if ( !v7 )
    return sub_CE9620(a1);
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v9 = *(unsigned __int8 **)(v7 + 24);
          v10 = *v9;
          if ( (_BYTE)v10 == 4 )
            goto LABEL_28;
          if ( !a3 )
            break;
          sub_E33C60(&v36, v7);
          if ( !v36 || !v38 )
          {
            if ( v37 != v39 )
              _libc_free(v37, v7);
            v10 = *v9;
            break;
          }
          if ( v37 == v39 )
            goto LABEL_28;
          _libc_free(v37, v7);
          v7 = *(_QWORD *)(v7 + 8);
          if ( !v7 )
            return sub_CE9620(a1);
        }
        if ( (unsigned __int8)v10 > 0x1Cu )
          break;
        if ( !a4 || (_BYTE)v10 != 5 || (unsigned __int16)(*((_WORD *)v9 + 1) - 49) > 1u )
          goto LABEL_6;
LABEL_33:
        v11 = *((_QWORD *)v9 + 2);
        if ( v11 )
        {
          v18 = *((_QWORD *)v9 + 2);
          while ( 1 )
          {
            v19 = *(_QWORD *)(v18 + 24);
            if ( *(_BYTE *)v19 != 85 )
              break;
            v30 = *(_QWORD *)(v19 - 32);
            if ( !v30
              || *(_BYTE *)v30
              || *(_QWORD *)(v30 + 24) != *(_QWORD *)(v19 + 80)
              || (*(_BYTE *)(v30 + 33) & 0x20) == 0 )
            {
              break;
            }
            v31 = *(_DWORD *)(v30 + 36);
            if ( v31 > 0xD3 )
            {
              if ( v31 != 324 )
              {
                if ( v31 > 0x144 )
                {
                  if ( v31 != 376 )
                    break;
                }
                else if ( v31 != 282 && v31 - 291 > 1 )
                {
                  break;
                }
              }
            }
            else if ( v31 > 0x9A )
            {
              if ( ((1LL << ((unsigned __int8)v31 + 101)) & 0x186000000000001LL) == 0 )
                break;
            }
            else if ( v31 != 11 && v31 - 68 > 3 )
            {
              break;
            }
            v18 = *(_QWORD *)(v18 + 8);
            if ( !v18 )
              goto LABEL_28;
          }
          if ( !a5 )
            goto LABEL_17;
LABEL_8:
          v12 = *(_QWORD *)(v11 + 24);
          if ( (unsigned __int8)v10 <= 0x1Cu )
          {
            if ( (_BYTE)v10 != 5 || (unsigned __int16)(*((_WORD *)v9 + 1) - 49) > 1u )
              goto LABEL_10;
          }
          else if ( (unsigned __int8)(v10 - 78) > 1u )
          {
            goto LABEL_10;
          }
          if ( !*(_QWORD *)(v11 + 8) )
          {
            v27 = *(_QWORD *)(v12 + 16);
            if ( v27 )
            {
              v12 = *(_QWORD *)(v27 + 24);
              v11 = v27;
            }
          }
LABEL_10:
          while ( *(_BYTE *)v12 == 3 )
          {
            if ( (*(_BYTE *)(v12 + 7) & 0x10) == 0 )
              break;
            v13 = sub_BD5D20(v12);
            if ( v14 != 18
              || *(_QWORD *)v13 ^ 0x6D6F632E6D766C6CLL | *(_QWORD *)(v13 + 8) ^ 0x73752E72656C6970LL
              || *(_WORD *)(v13 + 16) != 25701 )
            {
              v28 = sub_BD5D20(v12);
              if ( v29 != 9 || *(_QWORD *)v28 != 0x6573752E6D766C6CLL || *(_BYTE *)(v28 + 8) != 100 )
                break;
            }
            v11 = *(_QWORD *)(v11 + 8);
            if ( !v11 )
              goto LABEL_28;
            v12 = *(_QWORD *)(v11 + 24);
          }
          goto LABEL_17;
        }
LABEL_28:
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return sub_CE9620(a1);
      }
      if ( (unsigned __int8)(v10 - 34) > 0x33u || (v20 = 0x8000000000041LL, !_bittest64(&v20, (unsigned int)(v10 - 34))) )
      {
        if ( !a4 || (unsigned __int8)(v10 - 78) > 1u )
        {
LABEL_6:
          if ( !a5 )
            goto LABEL_17;
          v11 = *((_QWORD *)v9 + 2);
          if ( !v11 )
            goto LABEL_17;
          goto LABEL_8;
        }
        goto LABEL_33;
      }
      if ( !a4 )
        break;
      if ( (_BYTE)v10 != 85 )
        break;
      v16 = *((_QWORD *)v9 - 4);
      if ( !v16 || *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *((_QWORD *)v9 + 10) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
        break;
      v17 = *(_DWORD *)(v16 + 36);
      if ( v17 <= 0xD3 )
      {
        if ( v17 > 0x9A )
        {
          v32 = 0x186000000000001LL;
          if ( _bittest64(&v32, v17 - 155) )
            goto LABEL_28;
        }
        else if ( v17 == 11 || v17 - 68 <= 3 )
        {
          goto LABEL_28;
        }
        break;
      }
      if ( v17 == 324 )
        goto LABEL_28;
      if ( v17 > 0x144 )
      {
        if ( v17 != 376 )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return sub_CE9620(a1);
      }
      else
      {
        if ( v17 == 282 )
          goto LABEL_28;
        if ( v17 - 291 > 1 )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return sub_CE9620(a1);
      }
    }
    if ( (unsigned __int8 *)v7 == v9 - 32 && (a7 || *(_QWORD *)(a1 + 24) == *((_QWORD *)v9 + 10)) )
      goto LABEL_28;
    if ( !a6 )
      goto LABEL_17;
    v21 = sub_BD2910(v7);
    if ( (v9[7] & 0x80u) == 0 )
      goto LABEL_17;
    v22 = sub_BD2BC0(v9);
    v24 = v22 + v23;
    if ( (v9[7] & 0x80u) == 0 )
      break;
    if ( !(unsigned int)((v24 - sub_BD2BC0(v9)) >> 4) )
      goto LABEL_17;
    if ( (v9[7] & 0x80u) == 0 )
      goto LABEL_64;
    if ( v21 < *(_DWORD *)(sub_BD2BC0(v9) + 8) )
      goto LABEL_17;
    if ( (v9[7] & 0x80u) == 0 )
      BUG();
    v25 = sub_BD2BC0(v9);
    if ( v21 >= *(_DWORD *)(v25 + v26 - 4) || *(_DWORD *)(*(_QWORD *)sub_B49810(v9, v21) + 8LL) != 6 )
      goto LABEL_17;
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      return sub_CE9620(a1);
  }
  if ( (unsigned int)(v24 >> 4) )
LABEL_64:
    BUG();
LABEL_17:
  if ( a2 )
    *a2 = v9;
  return 1;
}
