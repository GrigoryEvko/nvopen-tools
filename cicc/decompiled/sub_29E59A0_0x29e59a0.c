// Function: sub_29E59A0
// Address: 0x29e59a0
//
char __fastcall sub_29E59A0(__int64 a1, char a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rdx
  char *v7; // rax
  char v8; // dl
  unsigned __int8 *v9; // rdi
  int v10; // edx
  unsigned __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int8 v17; // al
  __int64 v18; // rax
  _QWORD **v19; // rbx
  _QWORD *v20; // r14
  __int64 v21; // rdi
  unsigned __int8 v22; // al
  bool v23; // r8
  __int64 v24; // r10
  __int64 v25; // r11
  __int64 v26; // rsi
  __int64 i; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // rsi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rax
  __int64 *v36; // r13
  __int64 *v37; // r14
  __int64 v38; // rsi
  __int64 v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+18h] [rbp-88h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  __int64 v43; // [rsp+28h] [rbp-78h]
  __int64 v44; // [rsp+30h] [rbp-70h]
  _QWORD **v45; // [rsp+38h] [rbp-68h]
  _QWORD *v46[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v47; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v48; // [rsp+58h] [rbp-48h]
  __int64 *v49; // [rsp+60h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v41 = sub_B91C10(a1, 35);
    LOBYTE(v4) = v41 == 0;
  }
  else
  {
    v41 = 0;
    LOBYTE(v4) = 1;
  }
  if ( a2 == 1 || !(_BYTE)v4 )
  {
    LODWORD(v4) = *(_DWORD *)(a3 + 16);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD *)(a3 + 8);
      v4 = v5 + ((unsigned __int64)*(unsigned int *)(a3 + 24) << 6);
      v43 = v4;
      if ( v5 != v4 )
      {
        v6 = v5 + ((unsigned __int64)*(unsigned int *)(a3 + 24) << 6);
        while ( 1 )
        {
          v4 = *(_QWORD *)(v5 + 24);
          if ( v4 != -8192 && v4 != -4096 )
            break;
          v5 += 64;
          if ( v6 == v5 )
            return v4;
        }
        if ( v43 != v5 )
        {
          while ( 1 )
          {
            v7 = *(char **)(v5 + 24);
            if ( v7 )
            {
              v8 = *v7;
              if ( (unsigned __int8)*v7 <= 0x1Cu || (unsigned __int8)(v8 - 34) > 0x33u )
              {
                v7 = 0;
              }
              else if ( ((0x8000000000041uLL >> (v8 - 34)) & 1) == 0 )
              {
                v7 = 0;
              }
            }
            v9 = *(unsigned __int8 **)(v5 + 56);
            v44 = (__int64)v9;
            if ( v9 )
            {
              v10 = *v9;
              if ( (unsigned __int8)v10 > 0x1Cu )
              {
                v11 = (unsigned int)(v10 - 34);
                if ( (unsigned __int8)v11 <= 0x33u )
                {
                  v12 = 0x8000000000041LL;
                  if ( _bittest64(&v12, v11) )
                  {
                    if ( v7 )
                    {
                      if ( !v41 )
                      {
                        sub_B99FD0(v44, 0x22u, 0);
                        sub_B99FD0(v44, 0x23u, 0);
                        goto LABEL_67;
                      }
                      if ( (*(_BYTE *)(v44 + 7) & 0x20) != 0 )
                      {
                        v13 = sub_B91C10(v44, 35);
                        v14 = v13;
                        if ( v13 )
                        {
                          v14 = sub_BA72D0(v13, v41);
                          sub_B99FD0(v44, 0x23u, v14);
                        }
                        if ( (*(_BYTE *)(v44 + 7) & 0x20) != 0 )
                        {
                          v15 = sub_B91C10(v44, 34);
                          v42 = v15;
                          v16 = v15;
                          if ( v15 )
                            break;
                        }
                      }
                    }
                  }
                }
              }
            }
LABEL_67:
            LOBYTE(v4) = v43;
            v5 += 64;
            if ( v5 != v43 )
            {
              while ( 1 )
              {
                v4 = *(_QWORD *)(v5 + 24);
                if ( v4 != -8192 && v4 != -4096 )
                  break;
                v5 += 64;
                if ( v43 == v5 )
                  return v4;
              }
              if ( v43 != v5 )
                continue;
            }
            return v4;
          }
          v47 = 0;
          v48 = 0;
          v49 = 0;
          v17 = *(_BYTE *)(v15 - 16);
          if ( (v17 & 2) != 0 )
          {
            v18 = *(_QWORD *)(v16 - 32);
            v45 = (_QWORD **)(v18 + 8LL * *(unsigned int *)(v16 - 24));
            if ( v45 != (_QWORD **)v18 )
            {
LABEL_31:
              v40 = v5;
              v19 = (_QWORD **)v18;
              do
              {
                v20 = *v19;
                if ( (unsigned __int8)(*(_BYTE *)*v19 - 5) >= 0x20u )
                  v20 = 0;
                v21 = sub_10390E0((__int64)v20);
                v22 = *(_BYTE *)(v21 - 16);
                v23 = (v22 & 2) != 0;
                if ( (v22 & 2) != 0 )
                  v24 = *(_QWORD *)(v21 - 32);
                else
                  v24 = v21 - 8LL * ((v22 >> 2) & 0xF) - 16;
                if ( (*(_BYTE *)(v14 - 16) & 2) != 0 )
                  v25 = *(_QWORD *)(v14 - 32);
                else
                  v25 = v14 - 8LL * ((*(_BYTE *)(v14 - 16) >> 2) & 0xF) - 16;
                v26 = v25;
                for ( i = v24; ; i += 8 )
                {
                  if ( v23 )
                  {
                    if ( i == v24 + 8LL * *(unsigned int *)(v21 - 24) )
                      break;
                  }
                  else if ( i == v24 + 8LL * ((*(_WORD *)(v21 - 16) >> 6) & 0xF) )
                  {
                    break;
                  }
                  v28 = (*(_BYTE *)(v14 - 16) & 2) != 0
                      ? *(unsigned int *)(v14 - 24)
                      : (*(_WORD *)(v14 - 16) >> 6) & 0xFu;
                  if ( v26 == v25 + 8 * v28 )
                    break;
                  v29 = 0;
                  if ( **(_BYTE **)i == 1 )
                  {
                    v29 = *(_QWORD *)(*(_QWORD *)i + 136LL);
                    if ( *(_BYTE *)v29 != 17 )
                      v29 = 0;
                  }
                  v30 = 0;
                  if ( **(_BYTE **)v26 == 1 )
                  {
                    v30 = *(_QWORD *)(*(_QWORD *)v26 + 136LL);
                    if ( *(_BYTE *)v30 != 17 )
                      v30 = 0;
                  }
                  if ( *(_DWORD *)(v29 + 32) <= 0x40u )
                    v31 = *(_QWORD *)(v29 + 24);
                  else
                    v31 = **(_QWORD **)(v29 + 24);
                  if ( *(_DWORD *)(v30 + 32) <= 0x40u )
                    v32 = *(_QWORD *)(v30 + 24);
                  else
                    v32 = **(_QWORD **)(v30 + 24);
                  if ( v32 != v31 )
                    goto LABEL_61;
                  v26 += 8;
                }
                v46[0] = v20;
                v33 = v48;
                if ( v48 == v49 )
                {
                  sub_914280((__int64)&v47, v48, v46);
                }
                else
                {
                  if ( v48 )
                  {
                    *v48 = (__int64)v20;
                    v33 = v48;
                  }
                  v48 = v33 + 1;
                }
LABEL_61:
                ++v19;
              }
              while ( v45 != v19 );
              v34 = (unsigned __int64)v47;
              v5 = v40;
              if ( v48 != v47 )
              {
                v35 = v48 - v47;
                if ( (*(_BYTE *)(v42 - 16) & 2) != 0 )
                {
                  if ( v35 >= *(unsigned int *)(v42 - 24) )
                    goto LABEL_65;
LABEL_86:
                  sub_B99FD0(v44, 0x22u, 0);
                  v36 = v48;
                  v37 = v47;
                  v46[0] = 0;
                  v46[1] = 0;
                  while ( v36 != v37 )
                  {
                    v38 = *v37++;
                    sub_103A560((__int64)v46, v38);
                  }
                  if ( !(unsigned __int8)sub_1039FB0((__int64)v46, v44) )
                    sub_B99FD0(v44, 0x23u, 0);
                  sub_29E1290(v46[0]);
                  v34 = (unsigned __int64)v47;
                }
                else if ( v35 < ((*(_WORD *)(v42 - 16) >> 6) & 0xFu) )
                {
                  goto LABEL_86;
                }
LABEL_65:
                if ( !v34 )
                  goto LABEL_67;
LABEL_66:
                j_j___libc_free_0(v34);
                goto LABEL_67;
              }
            }
          }
          else
          {
            v18 = v42 - 8LL * ((v17 >> 2) & 0xF) - 16;
            v45 = (_QWORD **)(v18 + 8LL * ((*(_WORD *)(v42 - 16) >> 6) & 0xF));
            if ( v45 != (_QWORD **)v18 )
              goto LABEL_31;
          }
          sub_B99FD0(v44, 0x22u, 0);
          sub_B99FD0(v44, 0x23u, 0);
          v34 = (unsigned __int64)v47;
          if ( v47 )
            goto LABEL_66;
          goto LABEL_67;
        }
      }
    }
  }
  return v4;
}
