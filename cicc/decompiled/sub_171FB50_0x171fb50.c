// Function: sub_171FB50
// Address: 0x171fb50
//
__int64 __fastcall sub_171FB50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 *v6; // r13
  unsigned __int8 v7; // al
  char v8; // al
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r14
  char v16; // al
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // r15d
  __int64 v24; // rax
  __int64 v25; // r14
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // r15d
  __int64 v31; // rax
  __int64 v32; // r13
  char v33; // al
  __int64 v34; // r13
  __int64 v35; // r13
  __int64 v36; // rax
  int v37; // [rsp+Ch] [rbp-34h]
  int v38; // [rsp+Ch] [rbp-34h]

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 38 )
  {
    v6 = *(__int64 **)(a2 - 48);
    v7 = *((_BYTE *)v6 + 16);
    if ( v7 == 14 )
    {
      if ( (void *)v6[4] == sub_16982C0() )
      {
        v18 = v6[5];
        if ( (*(_BYTE *)(v18 + 26) & 7) != 3 )
          return 0;
        v9 = (__int64 *)(v18 + 8);
      }
      else
      {
        v8 = *((_BYTE *)v6 + 50);
        v9 = v6 + 4;
        if ( (v8 & 7) != 3 )
          return 0;
      }
      if ( (*((_BYTE *)v9 + 18) & 8) == 0 )
        return 0;
    }
    else
    {
      if ( *(_BYTE *)(*v6 + 8) != 16 || v7 > 0x10u )
        return 0;
      v11 = sub_15A1020(*(_BYTE **)(a2 - 48), a2, *v6, a4);
      v12 = v11;
      if ( v11 && *(_BYTE *)(v11 + 16) == 14 )
      {
        if ( *(void **)(v11 + 32) == sub_16982C0() )
        {
          v29 = *(_QWORD *)(v12 + 40);
          if ( (*(_BYTE *)(v29 + 26) & 7) != 3 )
            return 0;
          v13 = v29 + 8;
        }
        else
        {
          if ( (*(_BYTE *)(v12 + 50) & 7) != 3 )
            return 0;
          v13 = v12 + 32;
        }
        if ( (*(_BYTE *)(v13 + 18) & 8) == 0 )
          return 0;
      }
      else
      {
        v37 = *(_QWORD *)(*v6 + 32);
        if ( v37 )
        {
          v23 = 0;
          do
          {
            v24 = sub_15A0A60((__int64)v6, v23);
            v25 = v24;
            if ( !v24 )
              return 0;
            v26 = *(_BYTE *)(v24 + 16);
            if ( v26 != 9 )
            {
              if ( v26 != 14 )
                return 0;
              if ( *(void **)(v25 + 32) == sub_16982C0() )
              {
                v28 = *(_QWORD *)(v25 + 40);
                if ( (*(_BYTE *)(v28 + 26) & 7) != 3 )
                  return 0;
                v27 = v28 + 8;
              }
              else
              {
                if ( (*(_BYTE *)(v25 + 50) & 7) != 3 )
                  return 0;
                v27 = v25 + 32;
              }
              if ( (*(_BYTE *)(v27 + 18) & 8) == 0 )
                return 0;
            }
          }
          while ( v37 != ++v23 );
        }
      }
    }
    v10 = *(_QWORD *)(a2 - 24);
    if ( v10 )
      goto LABEL_10;
    return 0;
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 14 )
    return 0;
  v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v15 = *(_QWORD *)(a2 - 24 * v14);
  if ( *(_BYTE *)(v15 + 16) == 14 )
  {
    if ( *(void **)(v15 + 32) == sub_16982C0() )
    {
      v22 = *(_QWORD *)(v15 + 40);
      if ( (*(_BYTE *)(v22 + 26) & 7) != 3 )
        return 0;
      v17 = v22 + 8;
    }
    else
    {
      v16 = *(_BYTE *)(v15 + 50);
      v17 = v15 + 32;
      if ( (v16 & 7) != 3 )
        return 0;
    }
    if ( (*(_BYTE *)(v17 + 18) & 8) != 0 )
      goto LABEL_24;
    return 0;
  }
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 )
    return 0;
  v19 = sub_15A1020(*(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a2, 4 * v14, a4);
  v20 = v19;
  if ( v19 && *(_BYTE *)(v19 + 16) == 14 )
  {
    if ( *(void **)(v19 + 32) == sub_16982C0() )
    {
      v36 = *(_QWORD *)(v20 + 40);
      if ( (*(_BYTE *)(v36 + 26) & 7) != 3 )
        return 0;
      v21 = v36 + 8;
    }
    else
    {
      if ( (*(_BYTE *)(v20 + 50) & 7) != 3 )
        return 0;
      v21 = v20 + 32;
    }
    if ( (*(_BYTE *)(v21 + 18) & 8) == 0 )
      return 0;
  }
  else
  {
    v38 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
    if ( v38 )
    {
      v30 = 0;
      do
      {
        v31 = sub_15A0A60(v15, v30);
        v32 = v31;
        if ( !v31 )
          return 0;
        v33 = *(_BYTE *)(v31 + 16);
        if ( v33 != 9 )
        {
          if ( v33 != 14 )
            return 0;
          if ( *(void **)(v32 + 32) == sub_16982C0() )
          {
            v35 = *(_QWORD *)(v32 + 40);
            if ( (*(_BYTE *)(v35 + 26) & 7) != 3 )
              return 0;
            v34 = v35 + 8;
          }
          else
          {
            if ( (*(_BYTE *)(v32 + 50) & 7) != 3 )
              return 0;
            v34 = v32 + 32;
          }
          if ( (*(_BYTE *)(v34 + 18) & 8) == 0 )
            return 0;
        }
      }
      while ( v38 != ++v30 );
    }
  }
  v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
LABEL_24:
  v10 = *(_QWORD *)(a2 + 24 * (1 - v14));
  if ( !v10 )
    return 0;
LABEL_10:
  **(_QWORD **)(a1 + 8) = v10;
  return 1;
}
