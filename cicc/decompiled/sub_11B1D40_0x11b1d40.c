// Function: sub_11B1D40
// Address: 0x11b1d40
//
__int64 __fastcall sub_11B1D40(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned __int8 v3; // r15
  char v4; // r12
  int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 *v9; // rdx
  __int64 v10; // rax
  char v11; // r14
  unsigned __int8 **v12; // rdx
  unsigned __int8 *v13; // r14
  unsigned __int8 v14; // al
  _BYTE *v15; // r14
  bool v16; // al
  __int64 *v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  void **v21; // rax
  unsigned __int8 *v22; // rdx
  void **v23; // r14
  void **v24; // rax
  __int64 v25; // rdx
  void **v26; // rax
  void **v27; // r15
  void **v28; // r15
  unsigned int v29; // r14d
  _BYTE *v30; // rax
  _BYTE *v31; // r15
  char v32; // al
  void *v33; // rax
  _BYTE *v34; // r15
  unsigned int v35; // r15d
  void **v36; // rax
  void **v37; // rdx
  char v38; // al
  _BYTE *v39; // rdx
  unsigned __int8 *v40; // [rsp+0h] [rbp-40h]
  int v41; // [rsp+0h] [rbp-40h]
  int v42; // [rsp+0h] [rbp-40h]
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]
  unsigned __int8 *v46; // [rsp+8h] [rbp-38h]
  void **v47; // [rsp+8h] [rbp-38h]

  v2 = sub_920620(a2);
  v3 = *(_BYTE *)a2;
  v4 = v2 ^ 1 | (a2 == 0);
  if ( v4 )
    goto LABEL_5;
  if ( v3 > 0x1Cu )
  {
    v5 = v3 - 29;
    if ( v3 != 41 )
      goto LABEL_4;
LABEL_14:
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v9 = *(__int64 **)(a2 - 8);
    else
      v9 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v10 = *v9;
    if ( !*v9 )
      goto LABEL_5;
    goto LABEL_17;
  }
  v5 = *(unsigned __int16 *)(a2 + 2);
  if ( v5 == 12 )
    goto LABEL_14;
LABEL_4:
  if ( v5 != 16 )
    goto LABEL_5;
  v11 = *(_BYTE *)(a2 + 7) & 0x40;
  if ( (*(_BYTE *)(a2 + 1) & 0x10) != 0 )
  {
    if ( v11 )
      v12 = *(unsigned __int8 ***)(a2 - 8);
    else
      v12 = (unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v13 = *v12;
    v14 = **v12;
    if ( v14 == 18 )
    {
      if ( *((void **)v13 + 3) == sub_C33340() )
        v15 = (_BYTE *)*((_QWORD *)v13 + 4);
      else
        v15 = v13 + 24;
      v16 = (v15[20] & 7) == 3;
      goto LABEL_25;
    }
    v45 = *((_QWORD *)v13 + 1);
    v25 = (unsigned int)*(unsigned __int8 *)(v45 + 8) - 17;
    if ( (unsigned int)v25 <= 1 && v14 <= 0x15u )
    {
      v26 = (void **)sub_AD7630((__int64)v13, 0, v25);
      v27 = v26;
      if ( !v26 || *(_BYTE *)v26 != 18 )
      {
        if ( *(_BYTE *)(v45 + 8) == 17 )
        {
          v42 = *(_DWORD *)(v45 + 32);
          if ( v42 )
          {
            v35 = 0;
            while ( 1 )
            {
              v36 = (void **)sub_AD69F0(v13, v35);
              v37 = v36;
              if ( !v36 )
                break;
              v38 = *(_BYTE *)v36;
              v47 = v37;
              if ( v38 != 13 )
              {
                if ( v38 != 18 )
                  goto LABEL_37;
                v39 = v37[3] == sub_C33340() ? v47[4] : v47 + 3;
                if ( (v39[20] & 7) != 3 )
                  goto LABEL_37;
                v4 = 1;
              }
              if ( v42 == ++v35 )
                goto LABEL_72;
            }
          }
        }
        goto LABEL_37;
      }
      if ( v26[3] == sub_C33340() )
        v28 = (void **)v27[4];
      else
        v28 = v27 + 3;
      v16 = (*((_BYTE *)v28 + 20) & 7) == 3;
LABEL_25:
      if ( v16 )
      {
LABEL_26:
        v11 = *(_BYTE *)(a2 + 7) & 0x40;
        goto LABEL_34;
      }
      goto LABEL_37;
    }
  }
  else
  {
    if ( v11 )
      v17 = *(__int64 **)(a2 - 8);
    else
      v17 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v18 = *v17;
    if ( *(_BYTE *)v18 == 18 )
    {
      v43 = v18;
      if ( *(void **)(v18 + 24) == sub_C33340() )
      {
        v19 = *(_QWORD *)(v43 + 32);
        if ( (*(_BYTE *)(v19 + 20) & 7) != 3 )
          goto LABEL_5;
      }
      else
      {
        v19 = v43 + 24;
        if ( (*(_BYTE *)(v43 + 44) & 7) != 3 )
          goto LABEL_5;
      }
      if ( (*(_BYTE *)(v19 + 20) & 8) == 0 )
        goto LABEL_5;
LABEL_34:
      if ( v11 )
        v20 = *(_QWORD *)(a2 - 8);
      else
        v20 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v10 = *(_QWORD *)(v20 + 32);
      if ( !v10 )
        goto LABEL_37;
LABEL_17:
      **(_QWORD **)a1 = v10;
      return 1;
    }
    v44 = *(_QWORD *)(v18 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 <= 1 && *(_BYTE *)v18 <= 0x15u )
    {
      v40 = (unsigned __int8 *)v18;
      v21 = (void **)sub_AD7630(v18, 0, v18);
      v22 = v40;
      v23 = v21;
      if ( !v21 || *(_BYTE *)v21 != 18 )
      {
        if ( *(_BYTE *)(v44 + 8) == 17 )
        {
          v41 = *(_DWORD *)(v44 + 32);
          if ( v41 )
          {
            v29 = 0;
            while ( 1 )
            {
              v46 = v22;
              v30 = (_BYTE *)sub_AD69F0(v22, v29);
              v22 = v46;
              v31 = v30;
              if ( !v30 )
                break;
              v32 = *v30;
              if ( v32 != 13 )
              {
                if ( v32 != 18 )
                  goto LABEL_37;
                v33 = sub_C33340();
                v22 = v46;
                if ( *((void **)v31 + 3) == v33 )
                {
                  v34 = (_BYTE *)*((_QWORD *)v31 + 4);
                  if ( (v34[20] & 7) != 3 )
                    goto LABEL_37;
                }
                else
                {
                  if ( (v31[44] & 7) != 3 )
                    goto LABEL_37;
                  v34 = v31 + 24;
                }
                if ( (v34[20] & 8) == 0 )
                  goto LABEL_37;
                v4 = 1;
              }
              if ( v41 == ++v29 )
              {
LABEL_72:
                if ( v4 )
                  goto LABEL_26;
                goto LABEL_37;
              }
            }
          }
        }
        goto LABEL_37;
      }
      if ( v21[3] == sub_C33340() )
      {
        v24 = (void **)v23[4];
        if ( (*((_BYTE *)v24 + 20) & 7) != 3 )
          goto LABEL_37;
      }
      else
      {
        if ( (*((_BYTE *)v23 + 44) & 7) != 3 )
          goto LABEL_37;
        v24 = v23 + 3;
      }
      if ( (*((_BYTE *)v24 + 20) & 8) != 0 )
        goto LABEL_26;
LABEL_37:
      v3 = *(_BYTE *)a2;
    }
  }
LABEL_5:
  result = 0;
  if ( v3 == 85 )
  {
    v7 = *(_QWORD *)(a2 - 32);
    if ( v7 )
    {
      if ( !*(_BYTE *)v7 && *(_QWORD *)(v7 + 24) == *(_QWORD *)(a2 + 80) && *(_DWORD *)(v7 + 36) == *(_DWORD *)(a1 + 8) )
      {
        v8 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 16) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        if ( v8 )
        {
          **(_QWORD **)(a1 + 24) = v8;
          return 1;
        }
      }
    }
  }
  return result;
}
