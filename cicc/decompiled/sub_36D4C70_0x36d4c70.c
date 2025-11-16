// Function: sub_36D4C70
// Address: 0x36d4c70
//
__int64 __fastcall sub_36D4C70(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // r15
  _BYTE **v8; // rdx
  _BYTE *v9; // rdi
  unsigned __int8 v10; // r9
  unsigned __int8 v11; // al
  __int64 *v12; // rax
  __int64 v13; // rax
  _QWORD **v14; // rbx
  _QWORD **v15; // r12
  _QWORD *v16; // rdi
  unsigned __int8 **v18; // rdx
  unsigned __int8 *v19; // rdi
  __int64 *v20; // rax
  __int64 v21; // rax
  unsigned __int8 **v22; // rdx
  unsigned __int8 *v23; // rdi
  __int64 *v24; // rax
  __int64 *v25; // rax
  unsigned __int8 v26; // [rsp+17h] [rbp-39h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v26 = sub_BB98D0((_QWORD *)a1, a2);
  if ( v26 )
    return 0;
  *(_DWORD *)(a1 + 184) = 0;
  v3 = *(_QWORD *)(a2 + 80);
  v27 = a2 + 72;
  if ( v3 == a2 + 72 )
    return 0;
  do
  {
    if ( !v3 )
      BUG();
    v4 = *(_QWORD *)(v3 + 32);
    if ( v4 != v3 + 24 )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        if ( *(_BYTE *)(v4 - 24) != 85 )
          goto LABEL_6;
        v5 = *(_QWORD *)(v4 - 56);
        if ( !v5 || *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(v4 + 56) || (*(_BYTE *)(v5 + 33) & 0x20) == 0 )
          goto LABEL_6;
        v6 = *(_DWORD *)(v5 + 36);
        v7 = v4 - 24;
        if ( v6 == 8934 )
          break;
        if ( v6 == 8935 )
        {
          if ( (*(_BYTE *)(v4 - 17) & 0x40) != 0 )
            v18 = *(unsigned __int8 ***)(v4 - 32);
          else
            v18 = (unsigned __int8 **)(v7 - 32LL * (*(_DWORD *)(v4 - 20) & 0x7FFFFFF));
          v19 = *v18;
          while ( *v19 == 93 )
          {
            v19 = (unsigned __int8 *)*((_QWORD *)v19 - 4);
            if ( !v19 )
              BUG();
          }
          v10 = sub_CE8900(v19, a2);
          if ( v10 )
          {
LABEL_49:
            v26 = v10;
            v25 = (__int64 *)sub_BD5C60(v4 - 24);
            v13 = sub_ACD6D0(v25);
            goto LABEL_50;
          }
          if ( !(unsigned __int8)sub_CE8980(v19, a2)
            && !(unsigned __int8)sub_CE8A00(v19, a2)
            && !(unsigned __int8)sub_CE8830(v19) )
          {
            goto LABEL_6;
          }
LABEL_34:
          v20 = (__int64 *)sub_BD5C60(v4 - 24);
          v21 = sub_ACD720(v20);
          goto LABEL_35;
        }
        if ( v6 == 8933 )
        {
          if ( (*(_BYTE *)(v4 - 17) & 0x40) != 0 )
            v8 = *(_BYTE ***)(v4 - 32);
          else
            v8 = (_BYTE **)(v7 - 32LL * (*(_DWORD *)(v4 - 20) & 0x7FFFFFF));
          v9 = *v8;
          while ( *v9 == 93 )
          {
            v9 = (_BYTE *)*((_QWORD *)v9 - 4);
            if ( !v9 )
              BUG();
          }
          v10 = sub_CE8830(v9);
          if ( v10 )
            goto LABEL_49;
          v11 = sub_CE8A80(v9, a2);
          if ( v11 )
          {
            v26 = v11;
            v12 = (__int64 *)sub_BD5C60(v4 - 24);
            v13 = sub_ACD720(v12);
LABEL_50:
            a2 = v4 - 24;
            sub_36D4AB0(a1, v4 - 24, v13);
          }
        }
LABEL_6:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 + 24 == v4 )
          goto LABEL_23;
      }
      if ( (*(_BYTE *)(v4 - 17) & 0x40) != 0 )
        v22 = *(unsigned __int8 ***)(v4 - 32);
      else
        v22 = (unsigned __int8 **)(v7 - 32LL * (*(_DWORD *)(v4 - 20) & 0x7FFFFFF));
      v23 = *v22;
      while ( *v23 == 93 )
      {
        v23 = (unsigned __int8 *)*((_QWORD *)v23 - 4);
        if ( !v23 )
          BUG();
      }
      if ( !(unsigned __int8)sub_CE8A00(v23, a2) && !(unsigned __int8)sub_CE8980(v23, a2) )
      {
        if ( !(unsigned __int8)sub_CE8900(v23, a2) && !(unsigned __int8)sub_CE8830(v23) )
          goto LABEL_6;
        goto LABEL_34;
      }
      v24 = (__int64 *)sub_BD5C60(v4 - 24);
      v21 = sub_ACD6D0(v24);
LABEL_35:
      a2 = v4 - 24;
      sub_36D4AB0(a1, v4 - 24, v21);
      v26 = 1;
      goto LABEL_6;
    }
LABEL_23:
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v27 != v3 );
  v14 = *(_QWORD ***)(a1 + 176);
  v15 = &v14[*(unsigned int *)(a1 + 184)];
  while ( v15 != v14 )
  {
    v16 = *v14++;
    sub_B43D60(v16);
  }
  return v26;
}
