// Function: sub_1EE65F0
// Address: 0x1ee65f0
//
void __fastcall sub_1EE65F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  __int64 v6; // r10
  unsigned __int64 v7; // rbx
  __int16 v8; // ax
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // r12
  unsigned int v12; // esi
  unsigned __int8 v13; // al
  unsigned int v14; // edx
  char v15; // di
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 *v19; // r12
  __int64 v20; // rsi
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // r12
  unsigned int v24; // esi
  unsigned __int8 v25; // al
  char v26; // cl
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 *v29; // rbx
  __int64 *v30; // r12
  __int64 v31; // rsi
  unsigned int v32; // [rsp+Ch] [rbp-54h]
  _QWORD v33[3]; // [rsp+10h] [rbp-50h] BYREF
  char v34; // [rsp+28h] [rbp-38h]

  v6 = a1;
  v7 = a2;
  v33[0] = a1;
  v8 = *(_WORD *)(a2 + 46);
  v33[1] = a3;
  v33[2] = a4;
  v34 = a6;
  if ( !a5 )
  {
    if ( (v8 & 4) != 0 )
    {
      do
        v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v7 + 46) & 4) != 0 );
    }
    v21 = *(_QWORD *)(a2 + 24) + 24LL;
    do
    {
      v22 = *(_QWORD *)(v7 + 32);
      v23 = v22 + 40LL * *(unsigned int *)(v7 + 40);
      if ( v22 != v23 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v21 == v7 )
        break;
    }
    while ( (*(_BYTE *)(v7 + 46) & 4) != 0 );
    if ( v22 == v23 )
    {
LABEL_47:
      v29 = *(__int64 **)(v6 + 80);
      v30 = &v29[*(unsigned int *)(v6 + 88)];
      if ( v29 != v30 )
      {
        while ( 1 )
        {
          v31 = *v29++;
          sub_1EE5780(v6 + 160, v31);
          if ( v30 == v29 )
            break;
          v6 = v33[0];
        }
      }
      return;
    }
    while ( 1 )
    {
      if ( !*(_BYTE *)v22 )
      {
        v24 = *(_DWORD *)(v22 + 8);
        if ( v24 )
        {
          v25 = *(_BYTE *)(v22 + 3);
          v26 = *(_BYTE *)(v22 + 4) & 1;
          if ( (v25 & 0x10) != 0 )
          {
            if ( v26 || (*(_BYTE *)(v22 + 4) & 2) != 0 || (*(_DWORD *)v22 & 0xFFF00) == 0 )
            {
              if ( (((v25 & 0x10) != 0) & (v25 >> 6)) != 0 )
                goto LABEL_54;
            }
            else
            {
              v32 = *(_DWORD *)(v22 + 8);
              sub_1EE5AC0((__int64)v33, v24, v6);
              v6 = v33[0];
              v24 = v32;
              if ( (((*(_BYTE *)(v22 + 3) & 0x10) != 0) & (*(_BYTE *)(v22 + 3) >> 6)) != 0 )
              {
LABEL_54:
                if ( !v34 )
                {
                  sub_1EE5AC0((__int64)v33, v24, v6 + 160);
                  v6 = v33[0];
                }
                goto LABEL_41;
              }
            }
            sub_1EE5AC0((__int64)v33, v24, v6 + 80);
            v6 = v33[0];
          }
          else if ( !v26 && (*(_BYTE *)(v22 + 4) & 2) == 0 )
          {
            sub_1EE5AC0((__int64)v33, v24, v6);
            v6 = v33[0];
          }
        }
      }
LABEL_41:
      v27 = v22 + 40;
      v28 = v23;
      if ( v27 == v23 )
      {
        while ( 1 )
        {
          v7 = *(_QWORD *)(v7 + 8);
          if ( v21 == v7 || (*(_BYTE *)(v7 + 46) & 4) == 0 )
            break;
          v23 = *(_QWORD *)(v7 + 32);
          v28 = v23 + 40LL * *(unsigned int *)(v7 + 40);
          if ( v23 != v28 )
            goto LABEL_49;
        }
        v22 = v23;
        v23 = v28;
        if ( v22 == v28 )
          goto LABEL_47;
      }
      else
      {
        v23 = v27;
LABEL_49:
        v22 = v23;
        v23 = v28;
      }
    }
  }
  if ( (v8 & 4) != 0 )
  {
    do
      v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v7 + 46) & 4) != 0 );
  }
  v9 = *(_QWORD *)(a2 + 24) + 24LL;
  do
  {
    v10 = *(_QWORD *)(v7 + 32);
    v11 = v10 + 40LL * *(unsigned int *)(v7 + 40);
    if ( v10 != v11 )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( v9 == v7 )
      break;
  }
  while ( (*(_BYTE *)(v7 + 46) & 4) != 0 );
  if ( v10 != v11 )
  {
    do
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v10 )
        {
          v12 = *(_DWORD *)(v10 + 8);
          if ( v12 )
          {
            v13 = *(_BYTE *)(v10 + 3);
            v14 = (*(_DWORD *)v10 >> 8) & 0xFFF;
            v15 = *(_BYTE *)(v10 + 4) & 1;
            if ( (v13 & 0x10) != 0 )
            {
              if ( v15 )
                v14 = 0;
              if ( (((v13 & 0x10) != 0) & (v13 >> 6)) != 0 )
              {
                if ( !v34 )
                {
                  sub_1EE5920((__int64)v33, v12, v14, v6 + 160);
                  v6 = v33[0];
                }
              }
              else
              {
                sub_1EE5920((__int64)v33, v12, v14, v6 + 80);
                v6 = v33[0];
              }
            }
            else if ( !v15 && (*(_BYTE *)(v10 + 4) & 2) == 0 )
            {
              sub_1EE5920((__int64)v33, v12, v14, v6);
              v6 = v33[0];
            }
          }
        }
        v16 = v10 + 40;
        v17 = v11;
        if ( v16 == v11 )
          break;
        v11 = v16;
LABEL_21:
        v10 = v11;
        v11 = v17;
      }
      while ( 1 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v9 == v7 || (*(_BYTE *)(v7 + 46) & 4) == 0 )
          break;
        v11 = *(_QWORD *)(v7 + 32);
        v17 = v11 + 40LL * *(unsigned int *)(v7 + 40);
        if ( v11 != v17 )
          goto LABEL_21;
      }
      v10 = v11;
      v11 = v17;
    }
    while ( v10 != v17 );
  }
  v18 = *(__int64 **)(v6 + 80);
  v19 = &v18[*(unsigned int *)(v6 + 88)];
  if ( v18 != v19 )
  {
    while ( 1 )
    {
      v20 = *v18++;
      sub_1EE5780(v6 + 160, v20);
      if ( v19 == v18 )
        break;
      v6 = v33[0];
    }
  }
}
