// Function: sub_CB5DB0
// Address: 0xcb5db0
//
__int64 __fastcall sub_CB5DB0(__int64 a1, unsigned __int8 *a2, __int64 a3, char a4)
{
  unsigned __int8 *v5; // r12
  unsigned __int8 *v6; // r15
  __int64 v8; // rdi
  _BYTE *v9; // rax
  unsigned __int8 v10; // bl
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rax
  char v22; // si
  char *v23; // rax
  char *v24; // rax
  char v25; // si
  char *v26; // rax
  char v27; // si
  char *v28; // rax
  char v29; // si

  v5 = &a2[a3];
  if ( &a2[a3] != a2 )
  {
    v6 = a2;
    while ( 1 )
    {
      v10 = *v6;
      v11 = *(unsigned __int8 **)(a1 + 32);
      v12 = *(_QWORD *)(a1 + 24);
      if ( *v6 == 34 )
      {
        if ( v12 <= (unsigned __int64)v11 )
        {
          v18 = sub_CB5D20(a1, 92);
        }
        else
        {
          v18 = a1;
          *(_QWORD *)(a1 + 32) = v11 + 1;
          *v11 = 92;
        }
        v19 = *(_BYTE **)(v18 + 32);
        if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 24) )
        {
          sub_CB5D20(v18, 34);
        }
        else
        {
          *(_QWORD *)(v18 + 32) = v19 + 1;
          *v19 = 34;
        }
        goto LABEL_7;
      }
      if ( (char)v10 > 34 )
        break;
      if ( v10 == 9 )
      {
        if ( v12 <= (unsigned __int64)v11 )
        {
          v13 = sub_CB5D20(a1, 92);
        }
        else
        {
          v13 = a1;
          *(_QWORD *)(a1 + 32) = v11 + 1;
          *v11 = 92;
        }
        v14 = *(_BYTE **)(v13 + 32);
        if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 24) )
        {
          sub_CB5D20(v13, 116);
          goto LABEL_7;
        }
        ++v6;
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 116;
        if ( v5 == v6 )
          return a1;
      }
      else
      {
        if ( v10 == 10 )
        {
          if ( v12 <= (unsigned __int64)v11 )
          {
            v8 = sub_CB5D20(a1, 92);
            v9 = *(_BYTE **)(v8 + 32);
            if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
            {
LABEL_46:
              sub_CB5D20(v8, 110);
              goto LABEL_7;
            }
          }
          else
          {
            v8 = a1;
            *(_QWORD *)(a1 + 32) = v11 + 1;
            *v11 = 92;
            v9 = *(_BYTE **)(a1 + 32);
            if ( (unsigned __int64)v9 >= *(_QWORD *)(a1 + 24) )
              goto LABEL_46;
          }
          *(_QWORD *)(v8 + 32) = v9 + 1;
          *v9 = 110;
          goto LABEL_7;
        }
LABEL_25:
        if ( (unsigned __int8)(v10 - 32) <= 0x5Eu )
        {
          if ( v12 <= (unsigned __int64)v11 )
            goto LABEL_43;
LABEL_41:
          *(_QWORD *)(a1 + 32) = v11 + 1;
          *v11 = v10;
          goto LABEL_7;
        }
        if ( !a4 )
        {
          if ( v12 <= (unsigned __int64)v11 )
          {
            sub_CB5D20(a1, 92);
          }
          else
          {
            *(_QWORD *)(a1 + 32) = v11 + 1;
            *v11 = 92;
          }
          v26 = *(char **)(a1 + 32);
          v27 = (v10 >> 6) + 48;
          if ( (unsigned __int64)v26 >= *(_QWORD *)(a1 + 24) )
          {
            sub_CB5D20(a1, v27);
          }
          else
          {
            *(_QWORD *)(a1 + 32) = v26 + 1;
            *v26 = v27;
          }
          v28 = *(char **)(a1 + 32);
          v29 = ((v10 >> 3) & 7) + 48;
          if ( (unsigned __int64)v28 >= *(_QWORD *)(a1 + 24) )
          {
            sub_CB5D20(a1, v29);
          }
          else
          {
            *(_QWORD *)(a1 + 32) = v28 + 1;
            *v28 = v29;
          }
          v11 = *(unsigned __int8 **)(a1 + 32);
          v10 = (v10 & 7) + 48;
          if ( (unsigned __int64)v11 >= *(_QWORD *)(a1 + 24) )
          {
LABEL_43:
            v25 = v10;
LABEL_44:
            sub_CB5D20(a1, v25);
            goto LABEL_7;
          }
          goto LABEL_41;
        }
        if ( v12 <= (unsigned __int64)v11 )
        {
          v20 = sub_CB5D20(a1, 92);
          v21 = *(_BYTE **)(v20 + 32);
          if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 24) )
          {
LABEL_55:
            sub_CB5D20(v20, 120);
            goto LABEL_30;
          }
        }
        else
        {
          v20 = a1;
          *(_QWORD *)(a1 + 32) = v11 + 1;
          *v11 = 92;
          v21 = *(_BYTE **)(a1 + 32);
          if ( (unsigned __int64)v21 >= *(_QWORD *)(a1 + 24) )
            goto LABEL_55;
        }
        *(_QWORD *)(v20 + 32) = v21 + 1;
        *v21 = 120;
LABEL_30:
        v22 = a0123456789abcd_10[v10 >> 4];
        v23 = *(char **)(a1 + 32);
        if ( (unsigned __int64)v23 >= *(_QWORD *)(a1 + 24) )
        {
          sub_CB5D20(a1, v22);
        }
        else
        {
          *(_QWORD *)(a1 + 32) = v23 + 1;
          *v23 = v22;
        }
        v24 = *(char **)(a1 + 32);
        v25 = a0123456789abcd_10[v10 & 0xF];
        if ( (unsigned __int64)v24 >= *(_QWORD *)(a1 + 24) )
          goto LABEL_44;
        *(_QWORD *)(a1 + 32) = v24 + 1;
        *v24 = v25;
LABEL_7:
        if ( v5 == ++v6 )
          return a1;
      }
    }
    if ( v10 == 92 )
    {
      if ( v12 <= (unsigned __int64)v11 )
      {
        v16 = sub_CB5D20(a1, 92);
      }
      else
      {
        v16 = a1;
        *(_QWORD *)(a1 + 32) = v11 + 1;
        *v11 = 92;
      }
      v17 = *(_BYTE **)(v16 + 32);
      if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 24) )
      {
        sub_CB5D20(v16, 92);
      }
      else
      {
        *(_QWORD *)(v16 + 32) = v17 + 1;
        *v17 = 92;
      }
      goto LABEL_7;
    }
    goto LABEL_25;
  }
  return a1;
}
