// Function: sub_69D830
// Address: 0x69d830
//
__int64 __fastcall sub_69D830(__int64 a1, _DWORD *a2, __int64 a3, _QWORD *a4, __int64 **a5, _DWORD *a6, _DWORD *a7)
{
  __int64 v10; // rbx
  __int64 **v11; // rcx
  __int64 v12; // rsi
  char v13; // dl
  __int64 v14; // rax
  unsigned __int64 j; // rdx
  char v17; // al
  __int64 v18; // r13
  unsigned int v19; // r13d
  unsigned __int64 i; // rdx
  unsigned int v21; // edx
  __int64 ***v22; // rax
  int v23; // eax
  int v24; // eax
  __int64 *v25; // rax
  int v26; // eax
  __int64 **v27; // rax
  char v28; // al
  __int64 v29; // rax
  _DWORD *v30; // [rsp+0h] [rbp-40h]
  _DWORD *v31; // [rsp+0h] [rbp-40h]
  _DWORD *v32; // [rsp+0h] [rbp-40h]
  __int64 ***v33; // [rsp+0h] [rbp-40h]
  __int64 **v34; // [rsp+8h] [rbp-38h]
  __int64 **v35; // [rsp+8h] [rbp-38h]
  __int64 **v36; // [rsp+8h] [rbp-38h]
  _DWORD *v37; // [rsp+8h] [rbp-38h]

  v10 = a1;
  if ( a5 )
    *a5 = 0;
  v11 = (__int64 **)qword_4D03C50;
  v12 = dword_4F04C38;
  v13 = *(_BYTE *)(qword_4D03C50 + 17LL);
  if ( !dword_4F04C38 && (v13 & 0x40) == 0 )
    goto LABEL_5;
  v17 = *(_BYTE *)(a1 + 80);
  if ( v17 == 9 || v17 == 7 )
  {
    v18 = *(_QWORD *)(a1 + 88);
  }
  else
  {
    v18 = 0;
    if ( v17 == 21 )
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 192LL);
  }
  if ( (v13 & 2) == 0 && (*(_BYTE *)(v18 + 173) & 2) == 0 )
  {
    if ( (unsigned int)sub_693580() && *(_BYTE *)(a1 + 80) == 7 )
    {
      v12 = *qword_4F04C10;
      a1 = *((unsigned int *)qword_4F04C10 + 2);
      v11 = *(__int64 ***)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
      for ( i = (unsigned __int64)v11 >> 3; ; LODWORD(i) = v21 + 1 )
      {
        v21 = a1 & i;
        v22 = (__int64 ***)(v12 + 16LL * v21);
        a5 = *v22;
        if ( v11 == *v22 )
          break;
        if ( !a5 )
          goto LABEL_42;
      }
      a5 = v22[1];
LABEL_42:
      if ( a7 )
      {
        if ( ((_BYTE)a5[3] & 2) == 0 )
        {
          v12 = v18;
          a1 = (__int64)a5;
          if ( sub_68AF60(a5, v18) )
            *a7 = 1;
        }
      }
    }
    goto LABEL_5;
  }
  if ( *(_DWORD *)(a1 + 40) == unk_4F066A8
    || (*(_BYTE *)(a1 + 81) & 0x10) != 0
    || *(_QWORD *)(a1 + 64)
    || (v12 = *(_QWORD *)(v18 + 48)) == 0 )
  {
LABEL_5:
    v14 = unk_4F04C50;
LABEL_6:
    if ( !v14 )
      return 0;
    j = *(_QWORD *)(v14 + 32);
    if ( !j )
      return 0;
    goto LABEL_8;
  }
  j = *(_BYTE *)(qword_4D03C50 + 17LL) & 0x40;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) != 0 )
  {
LABEL_46:
    v19 = 394;
    goto LABEL_32;
  }
  v14 = unk_4F04C50;
  if ( unk_4F04C50 )
  {
    j = *(_QWORD *)(unk_4F04C50 + 32LL);
    if ( v12 == j )
    {
LABEL_8:
      if ( (*(_BYTE *)(j + 198) & 0x10) != 0
        && *(_BYTE *)(v10 + 80) == 7
        && (*(_BYTE *)(*(_QWORD *)(v10 + 88) + 172LL) & 2) != 0
        && (*(_DWORD *)(v10 + 40) == unk_4F066A8 || (*(_BYTE *)(v10 + 81) & 0x10) == 0 && *(_QWORD *)(v10 + 64)) )
      {
        v19 = 3561;
        goto LABEL_32;
      }
      return 0;
    }
  }
  if ( *(_BYTE *)(v18 + 136) <= 2u )
  {
    *(_BYTE *)(v18 + 170) |= 4u;
    goto LABEL_6;
  }
  if ( (*(_BYTE *)(v18 + 176) & 2) != 0 )
  {
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
    {
      if ( !a6 )
        goto LABEL_6;
      goto LABEL_52;
    }
    if ( a6 )
    {
      v30 = a6;
      v34 = a5;
      v23 = sub_693580();
      a6 = v30;
      if ( !v23 )
        goto LABEL_52;
      a5 = v34;
      if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 2) != 0 )
        goto LABEL_52;
    }
    else if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 2) != 0 )
    {
      goto LABEL_6;
    }
  }
  v31 = a6;
  v35 = a5;
  v24 = sub_693580();
  a5 = v35;
  a6 = v31;
  if ( !v24 )
  {
    if ( !dword_4D04964 || sub_6879B0() )
    {
      v28 = *(_BYTE *)(qword_4D03C50 + 17LL);
      if ( (v28 & 2) == 0 && ((v28 & 0x20) == 0 || dword_4F04C58 == -1) )
      {
        a1 = *(_QWORD *)(v18 + 120);
        if ( !(unsigned int)sub_8D4070(a1)
          && ((*(_BYTE *)(qword_4D03C50 + 18LL) & 2) == 0 || (*(_BYTE *)(v18 + 176) & 2) != 0) )
        {
          if ( !sub_6879B0() )
          {
            v12 = (__int64)a2;
            a1 = 394;
            sub_69D070(0x18Au, a2);
          }
          goto LABEL_5;
        }
      }
    }
    goto LABEL_46;
  }
  v12 = *qword_4F04C10;
  v11 = *(__int64 ***)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
  for ( j = (unsigned __int64)v11 >> 3; ; LODWORD(j) = j + 1 )
  {
    j = (_DWORD)qword_4F04C10[1] & (unsigned int)j;
    v25 = (__int64 *)(v12 + 16LL * (unsigned int)j);
    a1 = *v25;
    if ( v11 == (__int64 **)*v25 )
      break;
    if ( !a1 )
      goto LABEL_60;
  }
  a1 = v25[1];
LABEL_60:
  if ( a7 )
  {
    if ( (*(_BYTE *)(a1 + 24) & 2) == 0 )
    {
      v12 = v18;
      if ( sub_68AF60((__int64 **)a1, v18) )
        *a7 = 1;
    }
  }
  v32 = a6;
  v36 = a5;
  if ( a5 )
  {
    v26 = sub_8D2FB0(*(_QWORD *)(v18 + 120));
    a5 = v36;
    a6 = v32;
    if ( !v26 || (a1 = v18, v29 = sub_6EA7C0(v18, v12), a5 = v36, a6 = v32, !v29) || *(_BYTE *)(v29 + 173) != 6 )
    {
      v12 = (__int64)a2;
      a1 = v18;
      v33 = (__int64 ***)a5;
      v37 = a6;
      v27 = sub_5F7420((__int64 *)v18, a2, a6, 0);
      a5 = (__int64 **)v33;
      a6 = v37;
      *v33 = v27;
      if ( !v27 && !*v37 )
        goto LABEL_34;
      goto LABEL_5;
    }
    *v36 = 0;
LABEL_52:
    *a6 = 1;
    v14 = unk_4F04C50;
    goto LABEL_6;
  }
  v19 = 1734;
LABEL_32:
  if ( (unsigned int)sub_6E5430(a1, v12, j, v11, a5, a6) )
    sub_6851C0(v19, a2);
LABEL_34:
  sub_6E6260(a3);
  sub_6E5930(*a4);
  *a4 = 0;
  return 1;
}
