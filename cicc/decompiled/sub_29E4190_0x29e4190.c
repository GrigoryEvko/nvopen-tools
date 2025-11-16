// Function: sub_29E4190
// Address: 0x29e4190
//
void __fastcall sub_29E4190(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r9
  __int64 v4; // r12
  __int64 v5; // r8
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r15
  __int64 j; // rbx
  char v16; // al
  unsigned __int8 **v17; // rax
  unsigned __int8 **v18; // rax
  unsigned __int8 *v19; // rax
  unsigned __int8 *v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rax
  char *v24; // r15
  char v25; // al
  __int64 v26; // rax
  _BYTE *v27; // r15
  bool v28; // al
  unsigned __int8 **v29; // rax
  unsigned __int8 *v30; // rax
  _BYTE *v31; // rdi
  bool v32; // al
  unsigned __int8 **v33; // rax
  int v34; // edx
  int v35; // r9d
  unsigned __int8 *v36; // rax
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  __int64 v39; // [rsp+10h] [rbp-70h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  __int64 v41; // [rsp+18h] [rbp-68h]
  __int64 v43; // [rsp+28h] [rbp-58h]
  __int64 i; // [rsp+28h] [rbp-58h]
  unsigned __int64 v45; // [rsp+30h] [rbp-50h] BYREF
  char v46; // [rsp+40h] [rbp-40h]

  sub_B2EE70((__int64)&v45, a1, 0);
  if ( !v46 )
    return;
  v3 = a2;
  v4 = v45;
  v5 = a1;
  if ( a2 >= 0 )
  {
    v6 = a2 + v45;
    if ( !a3 )
      goto LABEL_6;
    goto LABEL_4;
  }
  if ( -a2 <= v45 )
  {
    v6 = a2 + v45;
    if ( !a3 )
      goto LABEL_12;
LABEL_4:
    v7 = v45 - v6;
    goto LABEL_5;
  }
  v6 = 0;
  if ( !a3 )
    goto LABEL_12;
  v7 = v45;
  v6 = 0;
LABEL_5:
  if ( *(_DWORD *)(a3 + 16) )
  {
    v21 = *(_QWORD *)(a3 + 8);
    v22 = v21 + ((unsigned __int64)*(unsigned int *)(a3 + 24) << 6);
    if ( v21 != v22 )
    {
      while ( 1 )
      {
        v23 = *(_QWORD *)(v21 + 24);
        if ( v23 != -8192 && v23 != -4096 )
          break;
        v21 += 64;
        if ( v22 == v21 )
          goto LABEL_6;
      }
      while ( v22 != v21 )
      {
        v24 = *(char **)(v21 + 24);
        v25 = *v24;
        if ( *v24 == 85 )
        {
          v31 = *(_BYTE **)(v21 + 56);
          if ( !v31 || *v31 != 85 )
            goto LABEL_45;
          v37 = v3;
          v39 = v5;
          sub_B4A9A0((__int64)v31, v7, v4);
          v32 = sub_B491E0((__int64)v31);
          v5 = v39;
          v3 = v37;
          if ( v32
            && (v33 = (unsigned __int8 **)*((_QWORD *)v31 - 4), *(_BYTE *)v33 == 61)
            && (v36 = sub_BD4070(*(v33 - 4), v7), v5 = v39, v3 = v37, v36)
            && *v36 > 0x1Cu )
          {
            sub_BC8F20(v36, v7, v4);
            v25 = *v24;
            v3 = v37;
            v5 = v39;
          }
          else
          {
            v25 = *v24;
          }
        }
        if ( v25 == 34 )
        {
          v27 = *(_BYTE **)(v21 + 56);
          if ( v27 )
          {
            if ( *v27 == 34 )
            {
              v38 = v3;
              v41 = v5;
              sub_B4B110(*(_QWORD *)(v21 + 56), v7, v4);
              v28 = sub_B491E0((__int64)v27);
              v5 = v41;
              v3 = v38;
              if ( v28 )
              {
                v29 = (unsigned __int8 **)*((_QWORD *)v27 - 4);
                if ( *(_BYTE *)v29 == 61 )
                {
                  v30 = sub_BD4070(*(v29 - 4), v7);
                  v5 = v41;
                  v3 = v38;
                  if ( v30 )
                  {
                    if ( *v30 > 0x1Cu )
                    {
                      sub_BC8F20(v30, v7, v4);
                      v3 = v38;
                      v5 = v41;
                    }
                  }
                }
              }
            }
          }
        }
LABEL_45:
        v21 += 64;
        if ( v21 == v22 )
          break;
        while ( 1 )
        {
          v26 = *(_QWORD *)(v21 + 24);
          if ( v26 != -8192 && v26 != -4096 )
            break;
          v21 += 64;
          if ( v22 == v21 )
            goto LABEL_6;
        }
      }
    }
  }
LABEL_6:
  if ( !v3 )
    return;
LABEL_12:
  v43 = v5;
  sub_B2F560(v5, v6, 0, 0);
  v40 = v43 + 72;
  for ( i = *(_QWORD *)(v43 + 80); v40 != i; i = *(_QWORD *)(i + 8) )
  {
    v8 = i - 24;
    if ( !i )
      v8 = 0;
    if ( !a3 )
      goto LABEL_19;
    v9 = *(unsigned int *)(a3 + 24);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD *)(a3 + 8);
      v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v12 = v10 + ((unsigned __int64)v11 << 6);
      v13 = *(_QWORD *)(v12 + 24);
      if ( v8 != v13 )
      {
        v34 = 1;
        while ( v13 != -4096 )
        {
          v35 = v34 + 1;
          v11 = (v9 - 1) & (v34 + v11);
          v12 = v10 + ((unsigned __int64)v11 << 6);
          v13 = *(_QWORD *)(v12 + 24);
          if ( v8 == v13 )
            goto LABEL_18;
          v34 = v35;
        }
        continue;
      }
LABEL_18:
      if ( v12 != v10 + (v9 << 6) )
      {
LABEL_19:
        v14 = *(_QWORD *)(v8 + 56);
        for ( j = v8 + 48; j != v14; v14 = *(_QWORD *)(v14 + 8) )
        {
          if ( !v14 )
            BUG();
          v16 = *(_BYTE *)(v14 - 24);
          if ( v16 == 85 )
          {
            sub_B4A9A0(v14 - 24, v6, v4);
            if ( sub_B491E0(v14 - 24) )
            {
              v17 = *(unsigned __int8 ***)(v14 - 56);
              if ( *(_BYTE *)v17 == 61 )
              {
                v20 = sub_BD4070(*(v17 - 4), v6);
                if ( v20 )
                {
                  if ( *v20 > 0x1Cu )
                    sub_BC8F20(v20, v6, v4);
                }
              }
            }
            v16 = *(_BYTE *)(v14 - 24);
          }
          if ( v16 == 34 )
          {
            sub_B4B110(v14 - 24, v6, v4);
            if ( sub_B491E0(v14 - 24) )
            {
              v18 = *(unsigned __int8 ***)(v14 - 56);
              if ( *(_BYTE *)v18 == 61 )
              {
                v19 = sub_BD4070(*(v18 - 4), v6);
                if ( v19 )
                {
                  if ( *v19 > 0x1Cu )
                    sub_BC8F20(v19, v6, v4);
                }
              }
            }
          }
        }
      }
    }
  }
}
