// Function: sub_182BD00
// Address: 0x182bd00
//
__int64 sub_182BD00()
{
  __int64 v0; // rax
  __int64 v1; // r12
  unsigned __int64 v2; // rsi
  _QWORD *v3; // rax
  _DWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  char v7; // dl
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  _DWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  _DWORD *v14; // r8
  _DWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v19; // rax
  _DWORD *v20; // r8
  _DWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  char v24; // dl

  v0 = sub_22077B0(408);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FA99CC;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F0AE8;
    *(_DWORD *)(v0 + 24) = 3;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 168) = v0 + 184;
    *(_QWORD *)(v0 + 176) = 0;
    *(_BYTE *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 400) = 0;
    v2 = sub_16D5D50();
    v3 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v4 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v5 = v3[2];
          v6 = v3[3];
          if ( v2 <= v3[4] )
            break;
          v3 = (_QWORD *)v3[3];
          if ( !v6 )
            goto LABEL_7;
        }
        v4 = v3;
        v3 = (_QWORD *)v3[2];
      }
      while ( v5 );
LABEL_7:
      v7 = 0;
      if ( v4 != dword_4FA0208 && v2 >= *((_QWORD *)v4 + 4) )
      {
        v19 = *((_QWORD *)v4 + 7);
        v20 = v4 + 12;
        if ( v19 )
        {
          v21 = v4 + 12;
          do
          {
            while ( 1 )
            {
              v22 = *(_QWORD *)(v19 + 16);
              v23 = *(_QWORD *)(v19 + 24);
              if ( *(_DWORD *)(v19 + 32) >= dword_4FA9F28 )
                break;
              v19 = *(_QWORD *)(v19 + 24);
              if ( !v23 )
                goto LABEL_30;
            }
            v21 = (_DWORD *)v19;
            v19 = *(_QWORD *)(v19 + 16);
          }
          while ( v22 );
LABEL_30:
          v7 = 0;
          if ( v20 != v21 && dword_4FA9F28 >= v21[8] )
          {
            v7 = byte_4FA9FC0;
            if ( (int)v21[9] <= 0 )
              v7 = 0;
          }
        }
      }
    }
    else
    {
      v7 = 0;
    }
    *(_BYTE *)(v1 + 265) = v7;
    v8 = sub_16D5D50();
    v9 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v10 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v11 = v9[2];
          v12 = v9[3];
          if ( v8 <= v9[4] )
            break;
          v9 = (_QWORD *)v9[3];
          if ( !v12 )
            goto LABEL_14;
        }
        v10 = v9;
        v9 = (_QWORD *)v9[2];
      }
      while ( v11 );
LABEL_14:
      if ( v10 == dword_4FA0208 )
        goto LABEL_23;
      if ( v8 < *((_QWORD *)v10 + 4) )
        goto LABEL_23;
      v13 = *((_QWORD *)v10 + 7);
      v14 = v10 + 12;
      if ( !v13 )
        goto LABEL_23;
      v15 = v10 + 12;
      do
      {
        while ( 1 )
        {
          v16 = *(_QWORD *)(v13 + 16);
          v17 = *(_QWORD *)(v13 + 24);
          if ( *(_DWORD *)(v13 + 32) >= dword_4FA9AC8 )
            break;
          v13 = *(_QWORD *)(v13 + 24);
          if ( !v17 )
            goto LABEL_21;
        }
        v15 = (_DWORD *)v13;
        v13 = *(_QWORD *)(v13 + 16);
      }
      while ( v16 );
LABEL_21:
      if ( v14 == v15 || dword_4FA9AC8 < v15[8] )
      {
LABEL_23:
        *(_BYTE *)(v1 + 264) = 0;
      }
      else
      {
        v24 = byte_4FA9B60;
        if ( (int)v15[9] <= 0 )
          v24 = 0;
        *(_BYTE *)(v1 + 264) = v24;
      }
    }
    else
    {
      *(_BYTE *)(v1 + 264) = 0;
    }
  }
  return v1;
}
