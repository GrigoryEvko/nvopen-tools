// Function: sub_648CF0
// Address: 0x648cf0
//
__int64 __fastcall sub_648CF0(_BYTE *a1, __int64 a2)
{
  __int64 v3; // r14
  char v4; // al
  _QWORD *v5; // r13
  int v6; // r15d
  __int64 v7; // rdi
  char v8; // al
  char v9; // dl
  char v10; // al
  __int64 result; // rax
  __int64 v12; // r8
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rcx
  char v16; // dl
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  char v20; // al
  char v21; // cl
  __int64 v22; // rdi
  __int64 v23; // rdx
  char v24; // al

  v3 = *(_QWORD *)a1;
  v4 = a1[64];
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 18LL) & 2) != 0 )
  {
    if ( (v4 & 9) != 0 || !dword_4D04964 )
    {
      v5 = 0;
      goto LABEL_9;
    }
LABEL_27:
    sub_6854E0(755, *(_QWORD *)(v3 + 24));
    goto LABEL_39;
  }
  v5 = *(_QWORD **)(v3 + 32);
  if ( (v4 & 9) == 0 && dword_4D04964 )
    goto LABEL_27;
  if ( (v4 & 8) != 0 && !(unsigned int)sub_85ED80(*(_QWORD *)(v3 + 24), qword_4F04C68[0] + 776LL * *((int *)a1 + 10)) )
  {
    sub_6854E0(551, *(_QWORD *)(v3 + 24));
LABEL_39:
    *(_BYTE *)(*(_QWORD *)a1 + 17LL) |= 0x20u;
    result = *(_QWORD *)a1;
    *(_QWORD *)(*(_QWORD *)a1 + 24LL) = 0;
    *((_QWORD *)a1 + 1) = 0;
    *((_DWORD *)a1 + 20) = 0;
    *((_QWORD *)a1 + 4) = 0;
    *((_QWORD *)a1 + 2) = 0;
    return result;
  }
LABEL_9:
  if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4) == 8
    && (unsigned int)sub_8D2310(*((_QWORD *)a1 + 7)) )
  {
    a1[64] |= 2u;
  }
  v6 = -1;
  if ( v5 )
  {
    if ( (a1[64] & 1) != 0 )
    {
      sub_864360(v5, 0);
    }
    else
    {
      sub_864230(v5, 0);
      v6 = *((_DWORD *)a1 + 10);
      *((_DWORD *)a1 + 10) = dword_4F04C64;
    }
    a1[65] |= 2u;
  }
  sub_642710((__int64 *)a1, a2);
  v7 = *((_QWORD *)a1 + 1);
  if ( (a1[65] & 4) != 0 )
  {
    v8 = *(_BYTE *)(v7 + 80);
    if ( v8 == 16 )
    {
      v7 = **(_QWORD **)(v7 + 88);
      v8 = *(_BYTE *)(v7 + 80);
    }
    v9 = a1[44];
    if ( v8 != 24 )
    {
      if ( v8 != 17 )
        goto LABEL_19;
      goto LABEL_31;
    }
    v7 = *(_QWORD *)(v7 + 88);
  }
  if ( v7 )
  {
    v8 = *(_BYTE *)(v7 + 80);
    v9 = a1[44];
    if ( v8 != 17 )
    {
LABEL_19:
      if ( !*(_BYTE *)(a2 + 268) )
      {
        switch ( v8 )
        {
          case 11:
            v9 = *(_BYTE *)(*(_QWORD *)(v7 + 88) + 172LL);
            break;
          case 20:
            v9 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 88) + 176LL) + 172LL);
            break;
          case 7:
            v9 = *(_BYTE *)(*(_QWORD *)(v7 + 88) + 136LL);
            break;
          default:
            sub_721090(v7);
        }
      }
      v10 = a1[64];
      if ( v9 == 2 )
      {
        *((_DWORD *)a1 + 20) = 1;
        a1[44] = 2;
      }
      else
      {
        *((_DWORD *)a1 + 20) = 2;
        if ( *(_BYTE *)(v7 + 80) != 11 )
          return sub_6418E0((__int64)a1);
        if ( (v10 & 8) == 0 )
        {
          if ( (v10 & 1) != 0 )
            return sub_6418E0((__int64)a1);
LABEL_61:
          v17 = *(_QWORD *)(v7 + 96);
          if ( !v17 )
            goto LABEL_43;
          if ( !dword_4D04340 )
          {
            sub_648C10(v7, v3 + 8);
            goto LABEL_43;
          }
          if ( (*(_BYTE *)(v17 + 80) & 4) != 0 )
            goto LABEL_43;
          sub_6854C0(752, v3 + 8, v7);
          goto LABEL_35;
        }
        a1[44] = 0;
      }
      if ( (v10 & 1) != 0 || *(_BYTE *)(v7 + 80) != 11 )
        goto LABEL_43;
      goto LABEL_61;
    }
  }
LABEL_31:
  if ( a1[86] )
    goto LABEL_43;
  v12 = *(_QWORD *)(v3 + 24);
  v13 = *(_BYTE *)(v12 + 80);
  if ( v13 == 17 )
  {
    v14 = *(_QWORD *)(v12 + 88);
    if ( v14 )
    {
      v15 = 0;
      do
      {
        v16 = *(_BYTE *)(v14 + 80);
        if ( v16 == 11 || v16 == 20 )
        {
          if ( v15 )
            goto LABEL_87;
          v15 = v14;
        }
        v14 = *(_QWORD *)(v14 + 8);
      }
      while ( v14 );
      if ( !v15 )
        goto LABEL_66;
      v24 = *(_BYTE *)(v15 + 80);
      if ( v24 == 24 )
      {
        v12 = v15;
        goto LABEL_71;
      }
      v12 = v15;
      if ( v24 == 17 )
      {
LABEL_87:
        sub_6854C0(493, v3 + 8, v12);
        goto LABEL_35;
      }
      goto LABEL_34;
    }
    goto LABEL_66;
  }
  if ( v13 != 24 )
  {
LABEL_34:
    sub_6854C0(147, v3 + 8, v12);
    goto LABEL_35;
  }
LABEL_71:
  v20 = *(_BYTE *)(v12 + 82);
  if ( (*(_BYTE *)(v3 + 18) & 2) != 0 )
  {
    if ( (v20 & 8) != 0 )
    {
      v21 = *(_BYTE *)(v12 + 80);
      v22 = v12;
      v23 = 0;
      goto LABEL_77;
    }
    if ( (v20 & 4) != 0 )
      goto LABEL_84;
LABEL_66:
    v18 = v3 + 8;
    v19 = *(_QWORD *)(*(_QWORD *)v3 + 8LL);
    if ( !v5 )
    {
      sub_6851A0(282, v18, v19);
      goto LABEL_39;
    }
    sub_686A10(742, v18, v19, *v5);
    goto LABEL_36;
  }
  v23 = *(_QWORD *)(v3 + 32);
  v22 = v12;
  if ( (v20 & 8) == 0 )
    goto LABEL_79;
  v21 = *(_BYTE *)(v12 + 80);
  v22 = v12;
  if ( v21 == 16 )
  {
    v22 = **(_QWORD **)(v12 + 88);
    v21 = *(_BYTE *)(v22 + 80);
  }
LABEL_77:
  if ( v21 == 24 )
    v22 = *(_QWORD *)(v22 + 88);
LABEL_79:
  if ( (v20 & 4) != 0 )
  {
LABEL_84:
    sub_6854C0(266, v3 + 8, v12);
LABEL_35:
    if ( !v5 )
      goto LABEL_39;
LABEL_36:
    if ( (a1[64] & 1) != 0 )
    {
      sub_8645D0();
    }
    else
    {
      sub_8642D0();
      *((_DWORD *)a1 + 10) = v6;
    }
    a1[65] &= ~2u;
    goto LABEL_39;
  }
  if ( !v23 || !(unsigned int)sub_880800(v22, *(_QWORD *)(v23 + 128)) )
    goto LABEL_66;
LABEL_43:
  result = *((unsigned int *)a1 + 20);
  if ( (_DWORD)result )
    return sub_6418E0((__int64)a1);
  return result;
}
