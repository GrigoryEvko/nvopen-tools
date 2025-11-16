// Function: sub_250C680
// Address: 0x250c680
//
unsigned __int64 __fastcall sub_250C680(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  unsigned __int64 v3; // r15
  char v4; // al
  __int64 v5; // r12
  unsigned __int64 v6; // rbx
  char *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  char v10; // r13
  char *v11; // rdi
  unsigned __int8 *v12; // r8
  unsigned int v13; // ebx
  __int64 v14; // rbx
  __int64 v15; // r14
  unsigned int v16; // eax
  unsigned __int8 *v17; // rdi
  unsigned __int8 *v18; // rax
  __int64 v19; // rcx
  unsigned __int8 *v20; // r13
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // ebx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-C0h]
  unsigned __int8 *v29; // [rsp+18h] [rbp-A8h]
  __int64 *v30; // [rsp+30h] [rbp-90h]
  __int64 *v31; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v32; // [rsp+40h] [rbp-80h] BYREF
  char *v33; // [rsp+48h] [rbp-78h]
  unsigned int v34; // [rsp+50h] [rbp-70h]
  char v35; // [rsp+58h] [rbp-68h] BYREF
  __int64 *v36; // [rsp+60h] [rbp-60h] BYREF
  __int64 v37; // [rsp+68h] [rbp-58h]
  _BYTE v38[80]; // [rsp+70h] [rbp-50h] BYREF

  v1 = *a1;
  v2 = *a1 & 3;
  if ( v2 != 3 && v2 != 2 )
  {
    v3 = v1 & 0xFFFFFFFFFFFFFFFCLL;
    if ( (v1 & 0xFFFFFFFFFFFFFFFCLL) != 0 && *(_BYTE *)v3 == 22 )
      return v3;
  }
  v4 = sub_2509800(a1);
  if ( v4 == 6 )
  {
    LODWORD(v5) = *(_DWORD *)((v1 & 0xFFFFFFFFFFFFFFFCLL) + 32);
  }
  else
  {
    if ( v4 != 7 )
      return 0;
    v5 = (__int64)((v1 & 0xFFFFFFFFFFFFFFFCLL)
                 - (*(_QWORD *)((v1 & 0xFFFFFFFFFFFFFFFCLL) + 24)
                  - 32LL * (*(_DWORD *)(*(_QWORD *)((v1 & 0xFFFFFFFFFFFFFFFCLL) + 24) + 4LL) & 0x7FFFFFF))) >> 5;
  }
  if ( (int)v5 < 0 )
    return 0;
  v6 = v1 & 0xFFFFFFFFFFFFFFFCLL;
  v36 = (__int64 *)v38;
  v37 = 0x400000000LL;
  v29 = (unsigned __int8 *)v6;
  if ( v2 == 3 )
    v29 = *(unsigned __int8 **)(v6 + 24);
  v7 = (char *)&v36;
  sub_E33A00(v29, (__int64)&v36);
  v30 = &v36[(unsigned int)v37];
  if ( v36 != v30 )
  {
    v31 = v36;
    v10 = 0;
    v3 = 0;
    while ( 1 )
    {
      v7 = (char *)*v31;
      sub_E33C60((__int64 *)&v32, *v31);
      if ( v34 || sub_B491E0((__int64)v32) )
      {
        v9 = (__int64)v32;
        v11 = v33;
        v8 = *((_DWORD *)v32 + 1) & 0x7FFFFFF;
        v12 = *(unsigned __int8 **)&v32[32 * (*(unsigned int *)v33 - v8)];
        if ( !v12 )
          goto LABEL_35;
      }
      else
      {
        v12 = (unsigned __int8 *)*((_QWORD *)v32 - 4);
        if ( !v12 )
        {
          v11 = v33;
          goto LABEL_35;
        }
      }
      if ( *sub_BD3990(v12, (__int64)v7) )
        goto LABEL_34;
      v13 = v34;
      if ( !v34 )
      {
        if ( !sub_B491E0((__int64)v32) )
        {
          v7 = (char *)v32;
          v8 = *v32;
          if ( (_DWORD)v8 == 40 )
          {
            v7 = (char *)v32;
            v9 = 32LL * (unsigned int)sub_B491D0((__int64)v32);
          }
          else
          {
            v9 = 0;
            if ( (_DWORD)v8 != 85 )
            {
              if ( (_DWORD)v8 != 34 )
                BUG();
              v9 = 64;
            }
          }
          if ( v7[7] < 0 )
          {
            v28 = v9;
            v22 = sub_BD2BC0((__int64)v7);
            v9 = v28;
            v23 = v22 + v8;
            if ( v7[7] >= 0 )
            {
              if ( (unsigned int)(v23 >> 4) )
LABEL_78:
                BUG();
            }
            else
            {
              v24 = sub_BD2BC0((__int64)v7);
              v9 = v28;
              if ( (unsigned int)((v23 - v24) >> 4) )
              {
                if ( v7[7] >= 0 )
                  goto LABEL_78;
                v25 = *(_DWORD *)(sub_BD2BC0((__int64)v7) + 8);
                if ( v7[7] >= 0 )
                  BUG();
                v26 = sub_BD2BC0((__int64)v7);
                v9 = v28;
                v27 = 32LL * (unsigned int)(*(_DWORD *)(v26 + v8 - 4) - v25);
                goto LABEL_65;
              }
            }
          }
          v27 = 0;
LABEL_65:
          v14 = (32LL * (*((_DWORD *)v7 + 1) & 0x7FFFFFF) - 32 - v9 - v27) >> 5;
          goto LABEL_18;
        }
        v13 = v34;
      }
      LODWORD(v14) = v13 - 1;
LABEL_18:
      if ( !(_DWORD)v14 )
        goto LABEL_34;
      v15 = 0;
      while ( 1 )
      {
        v8 = v34;
        if ( v34 || sub_B491E0((__int64)v32) )
          v16 = *(_DWORD *)&v33[4 * v15 + 4];
        else
          v16 = v15;
        if ( (_DWORD)v5 == v16 )
          break;
LABEL_29:
        if ( ++v15 == (unsigned int)v14 )
          goto LABEL_34;
      }
      if ( !v10 )
      {
        if ( v34 || sub_B491E0((__int64)v32) )
        {
          v7 = (char *)v32;
          v17 = *(unsigned __int8 **)&v32[32
                                        * (*(unsigned int *)v33 - (unsigned __int64)(*((_DWORD *)v32 + 1) & 0x7FFFFFF))];
          if ( !v17 )
            goto LABEL_76;
        }
        else
        {
          v17 = (unsigned __int8 *)*((_QWORD *)v32 - 4);
          if ( !v17 )
LABEL_76:
            BUG();
        }
        v18 = sub_BD3990(v17, (__int64)v7);
        v20 = v18;
        if ( *v18 )
          goto LABEL_76;
        if ( (v18[2] & 1) != 0 )
          sub_B2C6D0((__int64)v18, (__int64)v7, v8, v19);
        v9 = *((_QWORD *)v20 + 12);
        v10 = 1;
        v3 = v9 + 40 * v15;
        goto LABEL_29;
      }
      v3 = 0;
LABEL_34:
      v11 = v33;
LABEL_35:
      if ( v11 != &v35 )
        _libc_free((unsigned __int64)v11);
      if ( v30 == ++v31 )
      {
        if ( v3 && v10 )
          goto LABEL_40;
        break;
      }
    }
  }
  v3 = *((_QWORD *)v29 - 4);
  if ( v3 )
  {
    if ( *(_BYTE *)v3 || *(_QWORD *)(v3 + 104) <= (unsigned __int64)(int)v5 )
    {
      v3 = 0;
    }
    else
    {
      if ( (*(_BYTE *)(v3 + 2) & 1) != 0 )
        sub_B2C6D0(*((_QWORD *)v29 - 4), (__int64)v7, v8, v9);
      v3 = *(_QWORD *)(v3 + 96) + 40LL * (int)v5;
    }
  }
LABEL_40:
  if ( v36 != (__int64 *)v38 )
    _libc_free((unsigned __int64)v36);
  return v3;
}
