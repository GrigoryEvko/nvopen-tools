// Function: sub_16ECFD0
// Address: 0x16ecfd0
//
char *__fastcall sub_16ECFD0(__int64 a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  int v7; // ebx
  _DWORD *v8; // r12
  unsigned __int64 v9; // rcx
  int v10; // r13d
  int v11; // r10d
  __int64 v12; // rax
  int v13; // r14d
  int v14; // ebx
  __int64 v15; // r13
  __int64 v16; // rax
  int v17; // eax
  int v18; // eax
  int v19; // r8d
  char *result; // rax
  char *v21; // r13
  int v22; // eax
  int v23; // eax
  int v24; // edi
  int v25; // eax
  int v26[2]; // [rsp+0h] [rbp-80h]
  int v28; // [rsp+10h] [rbp-70h]
  unsigned __int64 v29; // [rsp+10h] [rbp-70h]
  int v30; // [rsp+10h] [rbp-70h]
  unsigned __int64 v31; // [rsp+10h] [rbp-70h]
  unsigned __int64 v32; // [rsp+10h] [rbp-70h]
  unsigned __int64 v33; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+20h] [rbp-60h]
  unsigned __int64 v36; // [rsp+28h] [rbp-58h]
  char *v37; // [rsp+30h] [rbp-50h]
  char *v38; // [rsp+38h] [rbp-48h]
  char *v39; // [rsp+40h] [rbp-40h]
  int v40; // [rsp+4Ch] [rbp-34h]

  v6 = a5;
  v7 = 128;
  v39 = a2;
  v36 = *(_QWORD *)(a1 + 96);
  if ( *(char **)(a1 + 32) != a2 )
    v7 = *(a2 - 1);
  v8 = *(_DWORD **)a1;
  v38 = 0;
  v35 = 1LL << a5;
  v9 = sub_16EC790(*(_QWORD *)a1, a4, a5, 1LL << a4, 132, 1LL << a4);
  v37 = *(char **)(a1 + 40);
  if ( v37 == a2 )
    goto LABEL_31;
LABEL_4:
  v40 = *v39;
  if ( v7 != 10 )
  {
    if ( v7 != 128 )
    {
      if ( v40 != 10 )
      {
LABEL_35:
        v10 = 0;
        goto LABEL_36;
      }
      v10 = v8[10] & 8;
      if ( !v10 )
        goto LABEL_36;
      goto LABEL_8;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      if ( v40 != 10 || (v8[10] & 8) == 0 )
        goto LABEL_25;
LABEL_8:
      v11 = v8[20];
      goto LABEL_9;
    }
LABEL_49:
    v11 = v8[19];
    if ( v40 == 10 )
    {
      if ( (v8[10] & 8) != 0 )
      {
        v11 += v8[20];
        goto LABEL_52;
      }
      if ( v11 <= 0 )
        goto LABEL_15;
    }
    else if ( v11 <= 0 )
    {
      goto LABEL_14;
    }
    v10 = 129;
    goto LABEL_10;
  }
  v10 = v8[10] & 8;
  if ( v10 )
    goto LABEL_49;
  while ( 1 )
  {
    v31 = v9;
    v23 = isalnum((unsigned __int8)v7);
    v9 = v31;
    if ( v7 != 95 )
    {
      if ( v23 )
      {
LABEL_41:
        if ( v7 == 95 || v23 )
          goto LABEL_43;
        goto LABEL_25;
      }
      v24 = (unsigned __int8)v7;
      if ( v40 == 128 )
      {
LABEL_40:
        v32 = v9;
        v23 = isalnum(v24);
        v9 = v32;
        goto LABEL_41;
      }
      goto LABEL_16;
    }
LABEL_43:
    v19 = 134;
    if ( v10 == 130 )
      goto LABEL_24;
    v7 = 128;
    if ( v40 != 128 )
    {
      v33 = v9;
      v25 = isalnum((unsigned __int8)v40);
      v9 = v33;
      if ( v40 != 95 )
      {
        v19 = 134;
        if ( !v25 )
          goto LABEL_24;
      }
      goto LABEL_25;
    }
    while ( 2 )
    {
      result = v38;
      v21 = v39;
      if ( (v9 & v35) != 0 )
        result = v39;
      v38 = result;
      if ( v9 == v36 || v39 == a3 )
        return result;
      ++v39;
      v9 = sub_16EC790((__int64)v8, a4, v6, v9, v7, v36);
      if ( v37 != v21 + 1 )
        goto LABEL_4;
LABEL_31:
      if ( v7 == 10 )
      {
        v10 = v8[10] & 8;
        if ( !v10 )
        {
          if ( (*(_BYTE *)(a1 + 8) & 2) == 0 )
            goto LABEL_67;
LABEL_66:
          v40 = 128;
          goto LABEL_13;
        }
        v11 = v8[19];
        if ( (*(_DWORD *)(a1 + 8) & 2) != 0 )
        {
LABEL_72:
          v10 = 129;
          if ( v11 > 0 )
          {
            v40 = 128;
            goto LABEL_10;
          }
          goto LABEL_66;
        }
      }
      else
      {
        v22 = *(_DWORD *)(a1 + 8);
        if ( v7 != 128 || (v22 & 1) != 0 )
        {
          if ( (v22 & 2) != 0 )
          {
            v40 = 128;
            goto LABEL_35;
          }
LABEL_67:
          v40 = 128;
          v11 = v8[20];
LABEL_9:
          v10 = 130;
          if ( v11 <= 0 )
            goto LABEL_36;
LABEL_10:
          v12 = v6;
          v28 = v7;
          v13 = v10;
          v14 = v11;
          v15 = v12;
          do
          {
            v9 = sub_16EC790((__int64)v8, a4, v15, v9, v13, v9);
            --v14;
          }
          while ( v14 );
          v16 = v15;
          v7 = v28;
          v10 = v13;
          v6 = v16;
LABEL_13:
          if ( v10 != 129 )
            goto LABEL_36;
LABEL_14:
          if ( v40 == 128 )
          {
            v10 = 129;
LABEL_55:
            if ( v7 == 128 )
              goto LABEL_25;
            v24 = (unsigned __int8)v7;
            goto LABEL_40;
          }
LABEL_15:
          v10 = 129;
LABEL_16:
          v29 = v9;
          v17 = isalnum((unsigned __int8)v40);
          v9 = v29;
          if ( v40 == 95 || v17 )
          {
            if ( v7 == 128
              || (*(_QWORD *)v26 = v29, v30 = v17, v18 = isalnum((unsigned __int8)v7), v9 = *(_QWORD *)v26, !v18)
              && v7 != 95
              || v30
              || (v19 = 134, v40 == 95) )
            {
              v19 = 133;
            }
LABEL_24:
            v9 = sub_16EC790((__int64)v8, a4, v6, v9, v19, v9);
LABEL_25:
            v7 = v40;
            continue;
          }
          goto LABEL_55;
        }
        v11 = v8[19];
        if ( (v22 & 2) != 0 )
          goto LABEL_72;
      }
      break;
    }
    v40 = 128;
    v11 += v8[20];
LABEL_52:
    v10 = 131;
    if ( v11 > 0 )
      goto LABEL_10;
LABEL_36:
    if ( v7 == 128 )
      goto LABEL_25;
  }
}
