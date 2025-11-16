// Function: sub_2B17690
// Address: 0x2b17690
//
char __fastcall sub_2B17690(__int64 a1)
{
  unsigned __int8 v2; // bl
  char v4; // r14
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned int v11; // r15d
  __int64 v12; // r13
  __int64 v13; // rdi
  char v14; // al
  __int64 v15; // r15
  bool v16; // al
  __int64 v17; // rdx
  _BYTE *v18; // rax
  char v19; // cl
  unsigned int v20; // r15d
  unsigned int v21; // edx
  __int64 v22; // rax
  unsigned int v23; // edx
  char v24; // [rsp+0h] [rbp-40h]
  int v25; // [rsp+0h] [rbp-40h]
  unsigned int v26; // [rsp+4h] [rbp-3Ch]
  int v27; // [rsp+8h] [rbp-38h]
  __int64 v28; // [rsp+8h] [rbp-38h]
  int v29; // [rsp+8h] [rbp-38h]

  v2 = *(_BYTE *)a1;
  if ( (unsigned __int8)(*(_BYTE *)a1 - 82) <= 1u )
    return sub_B527F0(a1);
  if ( (unsigned int)v2 - 42 <= 0x11 )
  {
    v4 = sub_B46D50((unsigned __int8 *)a1);
    if ( v4 )
      return v4;
    if ( v2 != 44 )
      goto LABEL_6;
    if ( (unsigned __int8)sub_BD3660(a1, 64) )
    {
LABEL_14:
      v2 = *(_BYTE *)a1;
LABEL_6:
      if ( v2 != 45 || (unsigned __int8)sub_BD3660(a1, 64) )
        return v4;
      v5 = *(_QWORD **)(a1 + 16);
      if ( v5 )
      {
        while ( 1 )
        {
          v6 = v5[3];
          if ( *(_BYTE *)v6 != 85 )
            break;
          v7 = *(_QWORD *)(v6 - 32);
          if ( !v7
            || *(_BYTE *)v7
            || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v6 + 80)
            || *(_DWORD *)(v7 + 36) != 170
            || *v5 != *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)) )
          {
            break;
          }
          v5 = (_QWORD *)v5[1];
          if ( !v5 )
            return 1;
        }
        return 0;
      }
      return 1;
    }
    v8 = *(__int64 **)(a1 + 16);
    if ( !v8 )
      return 1;
    while ( 1 )
    {
      v12 = v8[3];
      v13 = *v8;
      v14 = *(_BYTE *)v12;
      if ( *(_BYTE *)v12 != 82 )
        goto LABEL_24;
      if ( v13 != *(_QWORD *)(v12 - 64) )
        goto LABEL_14;
      v15 = *(_QWORD *)(v12 - 32);
      if ( *(_BYTE *)v15 > 0x15u )
        goto LABEL_14;
      if ( sub_AC30F0(*(_QWORD *)(v12 - 32)) )
      {
LABEL_39:
        if ( (unsigned int)sub_B53900(v12) - 32 <= 1 )
          goto LABEL_34;
        goto LABEL_40;
      }
      if ( *(_BYTE *)v15 == 17 )
        break;
      v17 = *(_QWORD *)(v15 + 8);
      v28 = v17;
      if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 > 1 )
        goto LABEL_40;
      v18 = sub_AD7630(v15, 0, v17);
      v19 = 0;
      if ( v18 && *v18 == 17 )
      {
        v20 = *((_DWORD *)v18 + 8);
        if ( v20 > 0x40 )
        {
          v16 = v20 == (unsigned int)sub_C444A0((__int64)(v18 + 24));
          goto LABEL_46;
        }
        if ( !*((_QWORD *)v18 + 3) )
          goto LABEL_39;
      }
      else if ( *(_BYTE *)(v28 + 8) == 17 )
      {
        v21 = 0;
        v29 = *(_DWORD *)(v28 + 32);
        while ( v29 != v21 )
        {
          v24 = v19;
          v26 = v21;
          v22 = sub_AD69F0((unsigned __int8 *)v15, v21);
          if ( !v22 )
            goto LABEL_40;
          v23 = v26;
          v19 = v24;
          if ( *(_BYTE *)v22 != 13 )
          {
            if ( *(_BYTE *)v22 != 17 )
              goto LABEL_40;
            if ( *(_DWORD *)(v22 + 32) <= 0x40u )
            {
              if ( *(_QWORD *)(v22 + 24) )
                goto LABEL_40;
              v19 = 1;
            }
            else
            {
              v25 = *(_DWORD *)(v22 + 32);
              if ( v25 != (unsigned int)sub_C444A0(v22 + 24) )
                goto LABEL_40;
              v23 = v26;
              v19 = 1;
            }
          }
          v21 = v23 + 1;
        }
        if ( v19 )
          goto LABEL_39;
      }
LABEL_40:
      v12 = v8[3];
      v13 = *v8;
      v14 = *(_BYTE *)v12;
LABEL_24:
      if ( v14 != 85 )
        goto LABEL_14;
      v9 = *(_QWORD *)(v12 - 32);
      if ( !v9 )
        goto LABEL_14;
      if ( *(_BYTE *)v9 )
        goto LABEL_14;
      if ( *(_QWORD *)(v9 + 24) != *(_QWORD *)(v12 + 80) )
        goto LABEL_14;
      if ( *(_DWORD *)(v9 + 36) != 1 )
        goto LABEL_14;
      if ( *(_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF)) != v13 )
        goto LABEL_14;
      v10 = *(_QWORD *)(v12 + 32 * (1LL - (*(_DWORD *)(v12 + 4) & 0x7FFFFFF)));
      if ( *(_BYTE *)v10 != 17 )
        goto LABEL_14;
      if ( sub_B44900(v13) )
      {
        v11 = *(_DWORD *)(v10 + 32);
        if ( v11 <= 0x40 )
        {
          if ( *(_QWORD *)(v10 + 24) != 1 )
            goto LABEL_14;
        }
        else if ( (unsigned int)sub_C444A0(v10 + 24) != v11 - 1 )
        {
          goto LABEL_14;
        }
      }
LABEL_34:
      v8 = (__int64 *)v8[1];
      if ( !v8 )
        return 1;
    }
    if ( *(_DWORD *)(v15 + 32) <= 0x40u )
    {
      v16 = *(_QWORD *)(v15 + 24) == 0;
    }
    else
    {
      v27 = *(_DWORD *)(v15 + 32);
      v16 = v27 == (unsigned int)sub_C444A0(v15 + 24);
    }
LABEL_46:
    if ( v16 )
      goto LABEL_39;
    goto LABEL_40;
  }
  return sub_B46D50((unsigned __int8 *)a1);
}
