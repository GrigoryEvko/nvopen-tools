// Function: sub_8E9510
// Address: 0x8e9510
//
_BYTE *__fastcall sub_8E9510(unsigned __int8 *a1, __int64 a2, char a3, __int64 a4)
{
  unsigned __int8 *v5; // r12
  int v7; // ebx
  unsigned __int8 v8; // al
  __int64 v9; // r9
  _BYTE *v10; // r14
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // rax
  char *v15; // rdi
  char v16; // al
  int v17; // r14d
  _BYTE *v18; // r8
  __int64 v19; // rax
  char v20; // al
  char v21; // al
  bool v22; // zf
  char *v23; // rdi
  unsigned __int8 *v24; // r9
  _QWORD *v25; // r11
  char *v26; // rdi
  unsigned __int8 *v27; // r9
  _QWORD *v28; // r11
  char *v29; // rdi
  unsigned __int8 *v30; // r9
  char v31; // r14
  char v32; // r13
  unsigned __int8 *v33; // rax
  unsigned __int8 *v34; // r11
  _BYTE *v35; // r11
  unsigned __int64 v36; // r13
  unsigned __int8 *v37; // rax
  int v38; // eax
  int v39; // eax
  __int64 v40; // rcx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rdx
  unsigned __int8 *v43; // rax
  int v44; // eax
  char v45; // al
  __int64 v46; // [rsp+8h] [rbp-A8h]
  char *s; // [rsp+10h] [rbp-A0h]
  _BYTE *v48; // [rsp+18h] [rbp-98h]
  _BYTE *v49; // [rsp+18h] [rbp-98h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  __int64 v51; // [rsp+18h] [rbp-98h]
  _BYTE *v52; // [rsp+18h] [rbp-98h]
  _BYTE *v53; // [rsp+18h] [rbp-98h]
  _BYTE *v54; // [rsp+18h] [rbp-98h]
  _BYTE *v55; // [rsp+18h] [rbp-98h]
  _BYTE *v56; // [rsp+18h] [rbp-98h]
  __int64 v57; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int64 v58; // [rsp+30h] [rbp-80h] BYREF
  __int64 v59; // [rsp+38h] [rbp-78h] BYREF
  char v60[8]; // [rsp+40h] [rbp-70h] BYREF
  int v61; // [rsp+48h] [rbp-68h]
  __int64 v62; // [rsp+50h] [rbp-60h]

  v5 = a1;
  *(_QWORD *)a2 = 0;
  *(_DWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  if ( *a1 == 66 )
  {
    v12 = *(_QWORD *)(a4 + 32);
    if ( (a3 & 1) == 0 )
      *(_QWORD *)(a4 + 32) = ++v12;
    if ( !v12 )
      sub_8E5790("[static from ", a4);
    v5 = sub_8E5810(a1 + 1, (__int64 *)v60, a4);
    if ( *(__int64 *)v60 <= 0 )
    {
      v14 = *(_QWORD *)(a4 + 32);
      if ( !*(_DWORD *)(a4 + 24) )
      {
        ++v14;
        ++*(_QWORD *)(a4 + 48);
        *(_DWORD *)(a4 + 24) = 1;
        *(_QWORD *)(a4 + 32) = v14;
      }
    }
    else
    {
      v5 += *(_QWORD *)v60;
      v14 = *(_QWORD *)(a4 + 32);
    }
    if ( !v14 )
      sub_8E5790((unsigned __int8 *)"] ", a4);
    if ( !v13 )
      --*(_QWORD *)(a4 + 32);
  }
  v7 = a3 & 2;
  if ( !v7 )
    ++*(_QWORD *)(a4 + 32);
  v8 = *v5;
  if ( *v5 != 78 )
  {
    if ( v8 != 90 )
    {
      if ( v8 == 83 && v5[1] != 116 )
      {
        v10 = (_BYTE *)sub_8EC8F0((_DWORD)v5, 0, 0, 0, 0, 0, 0, a4);
      }
      else
      {
        v10 = (_BYTE *)sub_8EC360(v5, a2, a4);
        if ( *v10 != 73 )
          goto LABEL_9;
        if ( *(_QWORD *)(a4 + 48) )
          goto LABEL_36;
        sub_8E5DC0((__int64)v5, 0, 0, 0, a4, v9);
      }
      if ( *v10 != 73 )
        goto LABEL_9;
LABEL_36:
      if ( *(_QWORD *)(a4 + 64) )
        ++*(_QWORD *)(a4 + 32);
      v10 = sub_8E9020(v10, a4);
      if ( !*(_QWORD *)(a4 + 64) )
        goto LABEL_10;
LABEL_39:
      --*(_QWORD *)(a4 + 32);
      goto LABEL_10;
    }
    *(_QWORD *)a2 = 0;
    *(_DWORD *)(a2 + 8) = 0;
    v17 = unk_4F5F774;
    *(_QWORD *)(a2 + 16) = 0;
    if ( v17 )
      ++*(_QWORD *)(a4 + 32);
    v18 = (_BYTE *)sub_8E9250(v5 + 1, 1u, a4);
    if ( *v18 == 69 )
    {
      v19 = *(_QWORD *)(a4 + 32);
      ++v18;
    }
    else
    {
      v19 = *(_QWORD *)(a4 + 32);
      if ( !*(_DWORD *)(a4 + 24) )
      {
        ++v19;
        ++*(_QWORD *)(a4 + 48);
        *(_DWORD *)(a4 + 24) = 1;
        *(_QWORD *)(a4 + 32) = v19;
      }
    }
    if ( v19 )
    {
      v20 = *v18;
      if ( *v18 != 115 )
      {
LABEL_57:
        if ( v20 != 100 )
        {
LABEL_58:
          if ( *v18 == 85 && v18[1] == 110 && v18[2] == 118 )
          {
            v21 = v18[3];
            if ( v21 == 100 )
            {
              v45 = v18[4];
              if ( v45 == 108 )
              {
                v22 = (*(_QWORD *)(a4 + 32))-- == 1;
                if ( v22 )
                {
                  v56 = v18;
                  sub_8E5790("__nv_dl_wrapper_t<", a4);
                  v18 = v56;
                }
                v30 = v18 + 5;
                v32 = 0;
LABEL_160:
                v31 = 0;
                if ( !*(_QWORD *)(a4 + 32) )
                  sub_8E5790("__nv_dl_tag<", a4);
                goto LABEL_90;
              }
              if ( v45 == 116 && v18[5] == 108 )
              {
                v22 = (*(_QWORD *)(a4 + 32))-- == 1;
                if ( v22 )
                {
                  v55 = v18;
                  sub_8E5790("__nv_dl_wrapper_t<", a4);
                  v30 = v55 + 6;
                  v31 = 1;
                  v32 = 0;
                  if ( !*(_QWORD *)(a4 + 32) )
                    sub_8E5790("__nv_dl_trailing_return_tag<", a4);
                }
                else
                {
                  v31 = 1;
                  v30 = v18 + 6;
                  v32 = 0;
                }
LABEL_90:
                v33 = sub_8E5D20(v30, &v58, a4);
                v58 -= 2LL;
                s = (char *)v33;
                v50 = sub_8E9FF0(v33, 0, 0, 0, 1, a4);
                sub_8EB260(s, 0, 0, a4);
                if ( !*(_QWORD *)(a4 + 32) )
                  sub_8E5790(",(", a4);
                v22 = *(_QWORD *)(a4 + 32) == 0;
                *(_QWORD *)v60 = 0;
                v61 = 0;
                v62 = 0;
                if ( v22 )
                  sub_8E5790("& :: ", a4);
                v51 = sub_8E9510(v50, v60, 2, a4);
                if ( !*(_QWORD *)(a4 + 32) )
                  sub_8E5790("), ", a4);
                v34 = (unsigned __int8 *)v51;
                if ( v31 )
                {
                  v46 = sub_8E9FF0(v51, 0, 0, 0, 1, a4);
                  sub_8EB260(v51, 0, 0, a4);
                  v34 = (unsigned __int8 *)v46;
                  if ( !*(_QWORD *)(a4 + 32) )
                    sub_8E5790((unsigned __int8 *)", ", a4);
                }
                v10 = sub_8E5D20(v34, &v57, a4);
                sprintf(v60, "%lu", v57 - 2);
                if ( !*(_QWORD *)(a4 + 32) )
                  sub_8E5790((unsigned __int8 *)v60, a4);
                if ( *(_QWORD *)(a4 + 32) )
                {
                  if ( !v32 )
                  {
                    if ( !v58 )
                    {
LABEL_112:
                      ++*(_QWORD *)(a4 + 32);
                      goto LABEL_61;
                    }
                    goto LABEL_105;
                  }
                }
                else
                {
                  sub_8E5790((unsigned __int8 *)"> ", a4);
                  if ( !v32 )
                    goto LABEL_104;
                  if ( !*(_QWORD *)(a4 + 32) )
                    sub_8E5790((unsigned __int8 *)",", a4);
                }
                v10 = (_BYTE *)(sub_8EBA20(v10, 0, 3, a4) + 1);
LABEL_104:
                if ( !v58 )
                {
LABEL_110:
                  if ( !*(_QWORD *)(a4 + 32) )
                    sub_8E5790((unsigned __int8 *)">", a4);
                  goto LABEL_112;
                }
LABEL_105:
                v35 = v10;
                v36 = 0;
                while ( 1 )
                {
                  if ( !*(_QWORD *)(a4 + 32) )
                    sub_8E5790((unsigned __int8 *)",", a4);
                  v52 = v35;
                  ++v36;
                  v10 = (_BYTE *)sub_8E9FF0(v10, 0, 0, 0, 1, a4);
                  sub_8EB260(v52, 0, 0, a4);
                  if ( v58 <= v36 )
                    break;
                  v35 = v10;
                }
                goto LABEL_110;
              }
            }
            else if ( v21 == 104 && v18[4] == 100 && v18[5] == 108 )
            {
              v22 = (*(_QWORD *)(a4 + 32))-- == 1;
              if ( v22 )
              {
                v49 = v18;
                sub_8E5790("__nv_hdl_wrapper_t<", a4);
                v18 = v49;
              }
              v23 = "true,";
              v24 = sub_8E5D20(v18 + 6, &v59, a4);
              if ( v59 == 2 )
                v23 = "false,";
              if ( !*(_QWORD *)(a4 + 32) )
                sub_8E5790((unsigned __int8 *)v23, a4);
              v26 = "true,";
              v27 = sub_8E5D20(v24, v25, a4);
              if ( v59 == 2 )
                v26 = "false,";
              if ( !*(_QWORD *)(a4 + 32) )
                sub_8E5790((unsigned __int8 *)v26, a4);
              v29 = "true,";
              v30 = sub_8E5D20(v27, v28, a4);
              if ( v59 == 2 )
                v29 = "false,";
              if ( *(_QWORD *)(a4 + 32) )
              {
                v31 = 0;
                v32 = 1;
                goto LABEL_90;
              }
              v32 = 1;
              sub_8E5790((unsigned __int8 *)v29, a4);
              goto LABEL_160;
            }
          }
          v10 = (_BYTE *)sub_8E9510(v18, a2, 3, a4);
LABEL_61:
          if ( *(_DWORD *)(a4 + 24) || *v10 != 95 )
            goto LABEL_63;
          v59 = -1;
          v38 = (unsigned __int8)v10[1];
          if ( (unsigned int)(v38 - 48) > 9 )
          {
            if ( (_BYTE)v38 == 95 && (unsigned int)(unsigned __int8)v10[2] - 48 <= 9 )
            {
              v43 = sub_8E5810(v10 + 2, &v59, a4);
              v10 = v43;
              if ( *v43 == 95 )
              {
                v10 = v43 + 1;
                if ( v59 >= 0 )
                  goto LABEL_122;
                v44 = *(_DWORD *)(a4 + 24);
              }
              else
              {
                v59 = -1;
                v44 = *(_DWORD *)(a4 + 24);
              }
              if ( v44 )
                goto LABEL_63;
            }
          }
          else
          {
            v10 += 2;
            v39 = (char)v38 - 48;
            v59 = v39;
            if ( v39 >= 0 )
            {
LABEL_122:
              if ( !*(_QWORD *)(a4 + 32) )
                sub_8E5790((unsigned __int8 *)" (instance ", a4);
              sprintf(v60, "%ld", v59 + 2);
              if ( !*(_QWORD *)(a4 + 32) )
              {
                sub_8E5790((unsigned __int8 *)v60, a4);
                if ( !*(_QWORD *)(a4 + 32) )
                {
                  v40 = *(_QWORD *)(a4 + 8);
                  v41 = v40 + 1;
                  if ( !*(_DWORD *)(a4 + 28) )
                  {
                    v42 = *(_QWORD *)(a4 + 16);
                    if ( v42 > v41 )
                    {
                      *(_BYTE *)(*(_QWORD *)a4 + v40) = 41;
                      v41 = *(_QWORD *)(a4 + 8) + 1LL;
                    }
                    else
                    {
                      *(_DWORD *)(a4 + 28) = 1;
                      if ( v42 )
                      {
                        *(_BYTE *)(*(_QWORD *)a4 + v42 - 1) = 0;
                        v41 = *(_QWORD *)(a4 + 8) + 1LL;
                      }
                    }
                  }
                  *(_QWORD *)(a4 + 8) = v41;
                }
              }
LABEL_63:
              if ( !unk_4F5F774 )
                goto LABEL_10;
              goto LABEL_39;
            }
          }
          ++*(_QWORD *)(a4 + 32);
          ++*(_QWORD *)(a4 + 48);
          *(_DWORD *)(a4 + 24) = 1;
          goto LABEL_63;
        }
        v59 = -1;
        if ( v18[1] == 95 )
        {
          v18 += 2;
        }
        else
        {
          v37 = sub_8E5810(v18 + 1, &v59, a4);
          v18 = v37;
          if ( v59 < 0 || *v37 != 95 )
          {
            if ( !*(_DWORD *)(a4 + 24) )
            {
              ++*(_QWORD *)(a4 + 32);
              ++*(_QWORD *)(a4 + 48);
              *(_DWORD *)(a4 + 24) = 1;
            }
            goto LABEL_58;
          }
          v18 = v37 + 1;
        }
        if ( !*(_DWORD *)(a4 + 24) )
        {
          if ( !*(_QWORD *)(a4 + 32) )
          {
            v53 = v18;
            sub_8E5790("[default argument ", a4);
            v18 = v53;
          }
          v54 = v18;
          sprintf(v60, "%ld", v59 + 2);
          v18 = v54;
          if ( !*(_QWORD *)(a4 + 32) )
          {
            sub_8E5790((unsigned __int8 *)v60, a4);
            v18 = v54;
            if ( !*(_QWORD *)(a4 + 32) )
            {
              sub_8E5790(" (from end)]::", a4);
              v18 = v54;
            }
          }
        }
        goto LABEL_58;
      }
    }
    else
    {
      v48 = v18;
      sub_8E5790((unsigned __int8 *)"::", a4);
      v18 = v48;
      v20 = *v48;
      if ( *v48 != 115 )
        goto LABEL_57;
      if ( !*(_QWORD *)(a4 + 32) )
      {
        sub_8E5790((unsigned __int8 *)"string", a4);
        v18 = v48;
      }
    }
    v10 = v18 + 1;
    goto LABEL_61;
  }
  *(_QWORD *)a2 = 0;
  v15 = (char *)(v5 + 1);
  *(_DWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  while ( 1 )
  {
    v16 = *v15;
    if ( *v15 == 75 )
    {
      *(_DWORD *)(a2 + 4) |= 1u;
      goto LABEL_27;
    }
    if ( v16 != 86 )
      break;
    *(_DWORD *)(a2 + 4) |= 2u;
LABEL_27:
    ++v15;
  }
  switch ( v16 )
  {
    case 'r':
      *(_DWORD *)(a2 + 4) |= 4u;
      goto LABEL_27;
    case 'R':
      *(_DWORD *)(a2 + 8) = 1;
      LODWORD(v15) = (_DWORD)v15 + 1;
      break;
    case 'O':
      *(_DWORD *)(a2 + 8) = 2;
      LODWORD(v15) = (_DWORD)v15 + 1;
      break;
  }
  v10 = (_BYTE *)sub_8EC3E0((_DWORD)v15, 0, (unsigned int)v60, (unsigned int)&v59, (int)a2 + 16, 0, a4);
  if ( *v10 == 69 )
  {
    ++v10;
  }
  else if ( !*(_DWORD *)(a4 + 24) )
  {
    ++*(_QWORD *)(a4 + 32);
    ++*(_QWORD *)(a4 + 48);
    *(_DWORD *)(a4 + 24) = 1;
  }
  if ( !(_DWORD)v59 )
    *(_DWORD *)a2 = 1;
  if ( *(_DWORD *)v60 )
LABEL_9:
    *(_DWORD *)a2 = 1;
LABEL_10:
  if ( !v7 )
    --*(_QWORD *)(a4 + 32);
  return v10;
}
