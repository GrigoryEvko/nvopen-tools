// Function: sub_1803D30
// Address: 0x1803d30
//
_QWORD *sub_1803D30()
{
  _QWORD *v0; // rax
  _QWORD *v1; // r12
  char v2; // al
  unsigned __int64 v3; // rsi
  _QWORD *v4; // rax
  _DWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  char v8; // dl
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  _DWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  char v14; // dl
  __int64 v15; // rax
  _DWORD *v16; // r8
  _DWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v22; // rax
  _DWORD *v23; // r8
  _DWORD *v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // rdx

  v0 = (_QWORD *)sub_22077B0(800);
  v1 = v0;
  if ( v0 )
  {
    v0[1] = 0;
    v0[2] = &unk_4FA6BD4;
    v0[10] = v0 + 8;
    v0[11] = v0 + 8;
    v0[16] = v0 + 14;
    v0[17] = v0 + 14;
    *v0 = off_49F08F0;
    v0[21] = v0 + 23;
    v2 = byte_4FA8200;
    *((_DWORD *)v1 + 6) = 3;
    *((_BYTE *)v1 + 230) = v2;
    v1[4] = 0;
    v1[5] = 0;
    v1[6] = 0;
    *((_DWORD *)v1 + 16) = 0;
    v1[9] = 0;
    v1[12] = 0;
    *((_DWORD *)v1 + 28) = 0;
    v1[15] = 0;
    v1[18] = 0;
    *((_BYTE *)v1 + 152) = 0;
    v1[22] = 0;
    *((_BYTE *)v1 + 184) = 0;
    v1[25] = 0;
    v1[26] = 0;
    v1[27] = 0;
    v1[90] = 0;
    *((_BYTE *)v1 + 728) = 0;
    v1[92] = 0;
    v1[93] = 0;
    v1[94] = 0;
    *((_DWORD *)v1 + 190) = 0;
    v1[96] = 0;
    v1[97] = 0;
    v1[98] = 0;
    *((_DWORD *)v1 + 198) = 0;
    v3 = sub_16D5D50();
    v4 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v5 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v6 = v4[2];
          v7 = v4[3];
          if ( v3 <= v4[4] )
            break;
          v4 = (_QWORD *)v4[3];
          if ( !v7 )
            goto LABEL_7;
        }
        v5 = v4;
        v4 = (_QWORD *)v4[2];
      }
      while ( v6 );
LABEL_7:
      v8 = 0;
      if ( v5 != dword_4FA0208 && v3 >= *((_QWORD *)v5 + 4) )
      {
        v22 = *((_QWORD *)v5 + 7);
        v23 = v5 + 12;
        if ( v22 )
        {
          v24 = v5 + 12;
          do
          {
            while ( 1 )
            {
              v25 = *(_QWORD *)(v22 + 16);
              v26 = *(_QWORD *)(v22 + 24);
              if ( *(_DWORD *)(v22 + 32) >= dword_4FA8CC8 )
                break;
              v22 = *(_QWORD *)(v22 + 24);
              if ( !v26 )
                goto LABEL_30;
            }
            v24 = (_DWORD *)v22;
            v22 = *(_QWORD *)(v22 + 16);
          }
          while ( v25 );
LABEL_30:
          v8 = 0;
          if ( v23 != v24 && dword_4FA8CC8 >= v24[8] )
          {
            v8 = byte_4FA8D60;
            if ( (int)v24[9] <= 0 )
              v8 = 0;
          }
        }
      }
    }
    else
    {
      v8 = 0;
    }
    *((_BYTE *)v1 + 229) = v8;
    v9 = sub_16D5D50();
    v10 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v11 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v12 = v10[2];
          v13 = v10[3];
          if ( v9 <= v10[4] )
            break;
          v10 = (_QWORD *)v10[3];
          if ( !v13 )
            goto LABEL_14;
        }
        v11 = v10;
        v10 = (_QWORD *)v10[2];
      }
      while ( v12 );
LABEL_14:
      v14 = 0;
      if ( v11 != dword_4FA0208 && v9 >= *((_QWORD *)v11 + 4) )
      {
        v15 = *((_QWORD *)v11 + 7);
        v16 = v11 + 12;
        if ( v15 )
        {
          v17 = v11 + 12;
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
          v14 = 0;
          if ( v16 != v17 && dword_4FA8DA8 >= v17[8] )
          {
            v14 = byte_4FA8E40;
            if ( (int)v17[9] <= 0 )
              v14 = 0;
          }
        }
      }
    }
    else
    {
      v14 = 0;
    }
    *((_BYTE *)v1 + 228) = v14;
    v20 = sub_163A1D0();
    sub_1803C40(v20);
  }
  return v1;
}
