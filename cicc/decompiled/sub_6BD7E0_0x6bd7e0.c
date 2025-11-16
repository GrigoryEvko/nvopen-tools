// Function: sub_6BD7E0
// Address: 0x6bd7e0
//
void *__fastcall sub_6BD7E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  void *v7; // r15
  __int64 v8; // r14
  int v9; // eax
  char v10; // al
  __int64 v11; // rdx
  void *result; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rdi
  _DWORD *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdx
  char v22; // cl
  __int64 v23; // [rsp+0h] [rbp-40h]
  _BOOL4 v24; // [rsp+8h] [rbp-38h]
  bool v25; // [rsp+Fh] [rbp-31h]

  v6 = a1;
  if ( dword_4D04964 )
  {
    v24 = 1;
    v25 = 1;
  }
  else
  {
    v24 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) != 0;
    v25 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) != 0;
  }
  v7 = &loc_1000008;
  v8 = a3;
  if ( (_DWORD)a2 == 2 )
    LODWORD(v7) = 16777240;
  if ( !a3 )
  {
    a2 = 0;
    v8 = sub_6BA760(0, 0);
  }
  if ( !(unsigned int)sub_8D2600(a1) || *(_BYTE *)(v8 + 8) != 1 || *(_QWORD *)(v8 + 24) )
  {
    if ( !(unsigned int)sub_8D3EA0(a1) )
    {
LABEL_10:
      v9 = sub_8D3B80(v6);
      sub_839D30(v8, v6, 1, v24, !v25, (_DWORD)v7, 1, v9, 0, a4, 0, 0);
      goto LABEL_11;
    }
    if ( dword_4F077C4 != 2 || unk_4F07778 <= 202301 )
    {
      a2 = 3347;
      if ( (unsigned int)sub_6E53E0(5, 3347, dword_4F07508) )
      {
        a2 = (__int64)dword_4F07508;
        sub_684B30(0xD13u, dword_4F07508);
      }
    }
    if ( *(_BYTE *)(v8 + 8) == 1 )
    {
      v13 = *(_QWORD **)(v8 + 24);
      if ( v13 )
      {
        if ( !*((_BYTE *)v13 + 8) )
        {
          if ( !*v13 )
            goto LABEL_45;
          if ( *(_BYTE *)(*v13 + 8LL) == 3 && !sub_6BBB10(v13) )
          {
            v13 = *(_QWORD **)(v8 + 24);
LABEL_45:
            v23 = v13[3];
            if ( !(unsigned int)sub_8DBE70(*(_QWORD *)(v23 + 8)) )
              v6 = sub_6EEB30(*(_QWORD *)(v23 + 8), 0);
            goto LABEL_10;
          }
        }
      }
    }
    v14 = v8;
    v15 = (_DWORD *)sub_6E1A20(v8);
    if ( (unsigned int)sub_6E5430(v8, a2, v16, v17, v18, v19) )
    {
      v14 = 171;
      sub_6851C0(0xABu, v15);
    }
    v6 = sub_72C930(v14);
    goto LABEL_10;
  }
  sub_6E7080(a4, 0);
  sub_6F7220(a4, a1);
LABEL_11:
  v10 = *(_BYTE *)(a4 + 16);
  if ( v10 == 1 )
  {
    v20 = *(_QWORD *)(a4 + 144);
  }
  else
  {
    if ( v10 != 2 )
      goto LABEL_13;
    v20 = *(_QWORD *)(a4 + 288);
    if ( v20 )
    {
      *(_BYTE *)(v20 + 26) |= 1u;
      if ( *(_BYTE *)(v20 + 24) != 5 )
        goto LABEL_13;
      goto LABEL_30;
    }
    if ( *(_BYTE *)(a4 + 317) != 12 || *(_BYTE *)(a4 + 320) != 1 )
      goto LABEL_13;
    v20 = sub_72E9A0(a4 + 144);
  }
  if ( !v20 )
    goto LABEL_13;
  *(_BYTE *)(v20 + 26) |= 1u;
  if ( *(_BYTE *)(v20 + 24) != 5 )
    goto LABEL_13;
LABEL_30:
  v21 = *(_QWORD *)(v20 + 56);
  v22 = *(_BYTE *)(v21 + 48);
  *(_BYTE *)(v21 + 50) |= 0x50u;
  if ( (v22 & 0xFB) == 2 )
    *(_BYTE *)(*(_QWORD *)(v21 + 56) + 168LL) |= 0x20u;
  *(_QWORD *)(v20 + 64) = v6;
LABEL_13:
  v11 = *(_QWORD *)sub_6E1A60(v8);
  result = &unk_4F061D8;
  unk_4F061D8 = v11;
  if ( !a3 )
    return (void *)sub_6E1990(v8);
  return result;
}
