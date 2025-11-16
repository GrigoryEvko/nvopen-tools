// Function: sub_18025E0
// Address: 0x18025e0
//
_QWORD *sub_18025E0()
{
  _QWORD *v0; // rax
  _QWORD *v1; // r12
  char v2; // al
  char v3; // al
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _DWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  char v9; // dl
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rax
  _DWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  _DWORD *v16; // r8
  _DWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v21; // rax
  _DWORD *v22; // r8
  _DWORD *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx
  char v26; // dl

  v0 = (_QWORD *)sub_22077B0(384);
  v1 = v0;
  if ( v0 )
  {
    v0[1] = 0;
    v0[2] = &unk_4FA6BCC;
    v0[10] = v0 + 8;
    v0[11] = v0 + 8;
    v0[16] = v0 + 14;
    v0[17] = v0 + 14;
    *v0 = off_49F0848;
    v2 = byte_4FA7200;
    *((_DWORD *)v1 + 6) = 5;
    *((_BYTE *)v1 + 202) = v2;
    v3 = byte_4FA7120;
    v1[4] = 0;
    *((_BYTE *)v1 + 203) = v3;
    v1[28] = v1 + 30;
    v1[5] = 0;
    v1[6] = 0;
    *((_DWORD *)v1 + 16) = 0;
    v1[9] = 0;
    v1[12] = 0;
    *((_DWORD *)v1 + 28) = 0;
    v1[15] = 0;
    v1[18] = 0;
    *((_BYTE *)v1 + 152) = 0;
    *((_BYTE *)v1 + 160) = 0;
    v1[21] = 0;
    v1[22] = 0;
    v1[23] = 0;
    *((_DWORD *)v1 + 48) = 0;
    v1[29] = 0;
    *((_BYTE *)v1 + 240) = 0;
    v1[32] = 0;
    v1[33] = 0;
    v1[34] = 0;
    v1[46] = 0;
    v1[47] = 0;
    v4 = sub_16D5D50();
    v5 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v6 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v7 = v5[2];
          v8 = v5[3];
          if ( v4 <= v5[4] )
            break;
          v5 = (_QWORD *)v5[3];
          if ( !v8 )
            goto LABEL_7;
        }
        v6 = v5;
        v5 = (_QWORD *)v5[2];
      }
      while ( v7 );
LABEL_7:
      v9 = 0;
      if ( v6 != dword_4FA0208 && v4 >= *((_QWORD *)v6 + 4) )
      {
        v21 = *((_QWORD *)v6 + 7);
        v22 = v6 + 12;
        if ( v21 )
        {
          v23 = v6 + 12;
          do
          {
            while ( 1 )
            {
              v24 = *(_QWORD *)(v21 + 16);
              v25 = *(_QWORD *)(v21 + 24);
              if ( *(_DWORD *)(v21 + 32) >= dword_4FA8CC8 )
                break;
              v21 = *(_QWORD *)(v21 + 24);
              if ( !v25 )
                goto LABEL_30;
            }
            v23 = (_DWORD *)v21;
            v21 = *(_QWORD *)(v21 + 16);
          }
          while ( v24 );
LABEL_30:
          v9 = 0;
          if ( v22 != v23 && dword_4FA8CC8 >= v23[8] )
          {
            v9 = byte_4FA8D60;
            if ( (int)v23[9] <= 0 )
              v9 = 0;
          }
        }
      }
    }
    else
    {
      v9 = 0;
    }
    *((_BYTE *)v1 + 201) = v9;
    v10 = sub_16D5D50();
    v11 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v12 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v13 = v11[2];
          v14 = v11[3];
          if ( v10 <= v11[4] )
            break;
          v11 = (_QWORD *)v11[3];
          if ( !v14 )
            goto LABEL_14;
        }
        v12 = v11;
        v11 = (_QWORD *)v11[2];
      }
      while ( v13 );
LABEL_14:
      if ( v12 == dword_4FA0208 )
        goto LABEL_23;
      if ( v10 < *((_QWORD *)v12 + 4) )
        goto LABEL_23;
      v15 = *((_QWORD *)v12 + 7);
      v16 = v12 + 12;
      if ( !v15 )
        goto LABEL_23;
      v17 = v12 + 12;
      do
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)(v15 + 16);
          v19 = *(_QWORD *)(v15 + 24);
          if ( *(_DWORD *)(v15 + 32) >= dword_4FA8DA8 )
            break;
          v15 = *(_QWORD *)(v15 + 24);
          if ( !v19 )
            goto LABEL_21;
        }
        v17 = (_DWORD *)v15;
        v15 = *(_QWORD *)(v15 + 16);
      }
      while ( v18 );
LABEL_21:
      if ( v17 == v16 || dword_4FA8DA8 < v17[8] )
      {
LABEL_23:
        *((_BYTE *)v1 + 200) = 0;
      }
      else
      {
        v26 = byte_4FA8E40;
        if ( (int)v17[9] <= 0 )
          v26 = 0;
        *((_BYTE *)v1 + 200) = v26;
      }
    }
    else
    {
      *((_BYTE *)v1 + 200) = 0;
    }
  }
  return v1;
}
