// Function: sub_BED100
// Address: 0xbed100
//
void __fastcall sub_BED100(_BYTE *a1, char *a2, const char *a3)
{
  unsigned __int8 v5; // dl
  _BYTE *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  const char *v9; // rax
  __int64 v10; // r13
  _BYTE *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  _BYTE *v14; // rax
  int v15; // eax
  char v16; // dl
  int v17; // r13d
  unsigned int v18; // eax
  unsigned int v19; // edx
  unsigned __int8 v20; // si
  __int64 v21; // rax
  bool v22; // si
  const char *v23; // r8
  const char *i; // rdi
  __int64 v25; // rax
  const char *v26; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v27[4]; // [rsp+10h] [rbp-50h] BYREF
  char v28; // [rsp+30h] [rbp-30h]
  char v29; // [rsp+31h] [rbp-2Fh]

  v26 = a3;
  v5 = *(a3 - 16);
  if ( (v5 & 2) != 0 )
  {
    if ( *((_DWORD *)a3 - 6) > 1u )
    {
      v6 = (_BYTE *)**((_QWORD **)a3 - 4);
      if ( v6 )
        goto LABEL_4;
LABEL_10:
      v29 = 1;
      v9 = "first operand should not be null";
      goto LABEL_19;
    }
LABEL_18:
    v29 = 1;
    v9 = "!prof annotations should have no less than 2 operands";
    goto LABEL_19;
  }
  if ( ((*((_WORD *)a3 - 8) >> 6) & 0xFu) <= 1 )
    goto LABEL_18;
  v6 = *(_BYTE **)&a3[-8 * ((v5 >> 2) & 0xF) - 16];
  if ( !v6 )
    goto LABEL_10;
LABEL_4:
  if ( *v6 )
  {
    v10 = *(_QWORD *)a1;
    v29 = 1;
    v27[0] = "expected string with name of the !prof annotation";
    v28 = 3;
    if ( v10 )
    {
      sub_CA0E80(v27, v10);
      v11 = *(_BYTE **)(v10 + 32);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
      {
        sub_CB5D20(v10, 10);
      }
      else
      {
        *(_QWORD *)(v10 + 32) = v11 + 1;
        *v11 = 10;
      }
      v12 = *(_QWORD *)a1;
      a1[152] = 1;
      if ( v12 && v26 )
      {
        sub_A62C00(v26, v12, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
        v13 = *(_QWORD *)a1;
        v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
        if ( (unsigned __int64)v14 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
          sub_CB5D20(v13, 10);
        }
        else
        {
          *(_QWORD *)(v13 + 32) = v14 + 1;
          *v14 = 10;
        }
      }
    }
    else
    {
      a1[152] = 1;
    }
    return;
  }
  v7 = sub_B91420((__int64)v6);
  if ( v8 != 14
    || *(_QWORD *)v7 != 0x775F68636E617262LL
    || *(_DWORD *)(v7 + 8) != 1751607653
    || *(_WORD *)(v7 + 12) != 29556 )
  {
    return;
  }
  v15 = sub_BC8980((__int64)v26);
  v16 = *a2;
  v17 = v15;
  if ( *a2 == 34 )
  {
    if ( (unsigned int)(v15 - 1) <= 1 )
      goto LABEL_27;
    v29 = 1;
    v9 = "Wrong number of InvokeInst branch_weights operands";
LABEL_19:
    v27[0] = v9;
    v28 = 3;
    sub_BECB10(a1, (__int64)v27, &v26);
    return;
  }
  if ( v16 == 31 )
  {
    v18 = ((*((_DWORD *)a2 + 1) & 0x7FFFFFF) == 3) + 1;
  }
  else if ( v16 == 32 )
  {
    v18 = (*((_DWORD *)a2 + 1) & 0x7FFFFFFu) >> 1;
  }
  else
  {
    v18 = 1;
    if ( v16 != 85 )
    {
      if ( v16 == 33 )
      {
        v18 = (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 1;
      }
      else
      {
        v18 = 2;
        if ( v16 != 86 )
        {
          if ( v16 == 40 )
          {
            v18 = *((_DWORD *)a2 + 22) + 1;
          }
          else
          {
            v29 = 1;
            v28 = 3;
            v27[0] = "!prof branch_weights are not allowed for this instruction";
            sub_BECB10(a1, (__int64)v27, &v26);
            v18 = 0;
          }
        }
      }
    }
  }
  if ( v18 != v17 )
  {
    v29 = 1;
    v9 = "Wrong number of operands";
    goto LABEL_19;
  }
LABEL_27:
  v19 = sub_BC8810((__int64)v26);
  v20 = *(v26 - 16);
  v21 = 8LL * ((v20 >> 2) & 0xF);
  v22 = (v20 & 2) != 0;
  v23 = &v26[-v21 - 16];
  if ( !v22 )
    goto LABEL_34;
LABEL_28:
  if ( v19 < *((_DWORD *)v26 - 6) )
  {
    for ( i = (const char *)*((_QWORD *)v26 - 4); ; i = v23 )
    {
      v25 = *(_QWORD *)&i[8 * v19];
      if ( !v25 )
        break;
      if ( *(_BYTE *)v25 != 1 || **(_BYTE **)(v25 + 136) != 17 )
      {
        v29 = 1;
        v27[0] = "!prof brunch_weights operand is not a const int";
        v28 = 3;
        sub_BDBF70((__int64 *)a1, (__int64)v27);
        return;
      }
      ++v19;
      if ( v22 )
        goto LABEL_28;
LABEL_34:
      if ( v19 >= ((*((_WORD *)v26 - 8) >> 6) & 0xFu) )
        return;
    }
    v29 = 1;
    v9 = "second operand should not be null";
    goto LABEL_19;
  }
}
