// Function: sub_111F6D0
// Address: 0x111f6d0
//
__int64 __fastcall sub_111F6D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  unsigned int v4; // r13d
  __int64 v6; // rax
  _BYTE *v7; // rax
  char v8; // cl
  _BYTE *v9; // rax
  char v10; // dl
  __int64 v11; // rdx
  __int64 v12; // r14
  unsigned int v13; // eax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // r15d
  bool v18; // al
  __int64 v19; // r15
  _BYTE *v20; // rax
  bool v21; // dl
  unsigned int v22; // r15d
  _BYTE *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rdx
  int v33; // [rsp+0h] [rbp-40h]
  bool v34; // [rsp+7h] [rbp-39h]

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 || *(_QWORD *)(v3 + 8) || *(_BYTE *)v2 != 57 )
    return 0;
  v6 = *(_QWORD *)(v2 - 64);
  if ( !v6 )
    goto LABEL_11;
  *(_QWORD *)a1[1] = v6;
  v7 = *(_BYTE **)(v2 - 32);
  v8 = *v7;
  if ( *v7 != 42 )
  {
LABEL_9:
    if ( v8 != 59 )
    {
LABEL_10:
      if ( v8 == 44 )
      {
        v26 = *((_QWORD *)v7 - 8);
        if ( v26 )
        {
          *(_QWORD *)a1[6] = v26;
          if ( *((_QWORD *)v7 - 4) == *(_QWORD *)a1[7] )
            goto LABEL_18;
        }
      }
      goto LABEL_11;
    }
    v29 = *((_QWORD *)v7 - 8);
    if ( v29 )
    {
      *(_QWORD *)a1[4] = v29;
      v30 = *((_QWORD *)v7 - 4);
      if ( v30 == *(_QWORD *)a1[5] )
        goto LABEL_18;
      if ( !v30 )
      {
LABEL_60:
        v8 = *v7;
        goto LABEL_10;
      }
LABEL_59:
      *(_QWORD *)a1[4] = v30;
      if ( *((_QWORD *)v7 - 8) == *(_QWORD *)a1[5] )
        goto LABEL_18;
      goto LABEL_60;
    }
    v30 = *((_QWORD *)v7 - 4);
    if ( v30 )
      goto LABEL_59;
LABEL_11:
    v7 = *(_BYTE **)(v2 - 32);
    goto LABEL_12;
  }
  v24 = *((_QWORD *)v7 - 8);
  if ( v24 )
  {
    *(_QWORD *)a1[2] = v24;
    v25 = *((_QWORD *)v7 - 4);
    if ( v25 == *(_QWORD *)a1[3] )
      goto LABEL_18;
    if ( !v25 )
    {
LABEL_47:
      v8 = *v7;
      goto LABEL_9;
    }
LABEL_46:
    *(_QWORD *)a1[2] = v25;
    if ( *((_QWORD *)v7 - 8) == *(_QWORD *)a1[3] )
      goto LABEL_18;
    goto LABEL_47;
  }
  v25 = *((_QWORD *)v7 - 4);
  if ( v25 )
    goto LABEL_46;
LABEL_12:
  if ( !v7 )
    return 0;
  *(_QWORD *)a1[1] = v7;
  v9 = *(_BYTE **)(v2 - 64);
  v10 = *v9;
  if ( *v9 == 42 )
  {
    v27 = *((_QWORD *)v9 - 8);
    if ( v27 )
    {
      *(_QWORD *)a1[2] = v27;
      v28 = *((_QWORD *)v9 - 4);
      if ( v28 == *(_QWORD *)a1[3] )
        goto LABEL_18;
      if ( !v28 )
      {
LABEL_55:
        v10 = *v9;
        goto LABEL_14;
      }
    }
    else
    {
      v28 = *((_QWORD *)v9 - 4);
      if ( !v28 )
        return 0;
    }
    *(_QWORD *)a1[2] = v28;
    if ( *((_QWORD *)v9 - 8) == *(_QWORD *)a1[3] )
      goto LABEL_18;
    goto LABEL_55;
  }
LABEL_14:
  if ( v10 == 59 )
  {
    v31 = *((_QWORD *)v9 - 8);
    if ( v31 )
    {
      *(_QWORD *)a1[4] = v31;
      v32 = *((_QWORD *)v9 - 4);
      if ( v32 == *(_QWORD *)a1[5] )
        goto LABEL_18;
      if ( !v32 )
      {
LABEL_65:
        v10 = *v9;
        goto LABEL_15;
      }
    }
    else
    {
      v32 = *((_QWORD *)v9 - 4);
      if ( !v32 )
        return 0;
    }
    *(_QWORD *)a1[4] = v32;
    if ( *((_QWORD *)v9 - 8) == *(_QWORD *)a1[5] )
      goto LABEL_18;
    goto LABEL_65;
  }
LABEL_15:
  if ( v10 != 44 )
    return 0;
  v11 = *((_QWORD *)v9 - 8);
  if ( !v11 )
    return 0;
  *(_QWORD *)a1[6] = v11;
  if ( *((_QWORD *)v9 - 4) != *(_QWORD *)a1[7] )
    return 0;
LABEL_18:
  v12 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v12 > 0x15u )
    return 0;
  LOBYTE(v13) = sub_AC30F0(*(_QWORD *)(a2 - 32));
  v4 = v13;
  if ( (_BYTE)v13 )
  {
LABEL_20:
    v4 = 1;
    if ( *a1 )
    {
      v15 = sub_B53900(a2);
      v16 = *a1;
      *(_DWORD *)v16 = v15;
      *(_BYTE *)(v16 + 4) = BYTE4(v15);
    }
    return v4;
  }
  if ( *(_BYTE *)v12 == 17 )
  {
    v17 = *(_DWORD *)(v12 + 32);
    if ( v17 <= 0x40 )
      v18 = *(_QWORD *)(v12 + 24) == 0;
    else
      v18 = v17 == (unsigned int)sub_C444A0(v12 + 24);
    if ( v18 )
      goto LABEL_20;
  }
  else
  {
    v19 = *(_QWORD *)(v12 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 > 1 )
      return v4;
    v20 = sub_AD7630(v12, 0, v14);
    if ( v20 && *v20 == 17 )
    {
      v21 = sub_9867B0((__int64)(v20 + 24));
LABEL_31:
      if ( v21 )
        goto LABEL_20;
    }
    else if ( *(_BYTE *)(v19 + 8) == 17 )
    {
      v33 = *(_DWORD *)(v19 + 32);
      if ( v33 )
      {
        v21 = 0;
        v22 = 0;
        while ( 1 )
        {
          v34 = v21;
          v23 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v12, v22);
          if ( !v23 )
            break;
          v21 = v34;
          if ( *v23 != 13 )
          {
            if ( *v23 != 17 )
              break;
            v21 = sub_9867B0((__int64)(v23 + 24));
            if ( !v21 )
              break;
          }
          if ( v33 == ++v22 )
            goto LABEL_31;
        }
      }
    }
  }
  return v4;
}
