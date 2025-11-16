// Function: sub_67BDC0
// Address: 0x67bdc0
//
__int64 __fastcall sub_67BDC0(unsigned int a1, int *a2)
{
  __int64 result; // rax
  _BYTE *v5; // r8
  __int64 v6; // rax
  int v7; // edi
  __int64 v8; // rdx
  unsigned int v9; // r15d
  __int64 v10; // rsi
  unsigned int v11; // r14d
  int v12; // eax
  __int64 v13; // r13
  char *v14; // r15
  __int64 v15; // r14
  int v16; // ecx
  __int64 v17; // r14
  char *v18; // r15
  __int64 v19; // rax
  int v20; // eax
  int v21; // [rsp+Ch] [rbp-54h]
  unsigned int v22; // [rsp+10h] [rbp-50h] BYREF
  int v23; // [rsp+14h] [rbp-4Ch] BYREF
  __int64 v24; // [rsp+18h] [rbp-48h] BYREF
  char v25[12]; // [rsp+20h] [rbp-40h] BYREF
  int v26; // [rsp+2Ch] [rbp-34h]

  if ( dword_4CFDEAC && dword_4CFDEA8 == a1 )
  {
    result = (unsigned int)dword_4CFDEA0;
    if ( dword_4CFDEA0 )
      *a2 = dword_4CFDEA4;
  }
  else
  {
    *a2 = 0;
    dword_4CFDEAC = 0;
    sub_729E70(a1, &v24, &v22, &v23);
    if ( !v22 )
      goto LABEL_5;
    if ( *(_QWORD *)(v24 + 64) )
      goto LABEL_5;
    if ( v23 )
      goto LABEL_5;
    v5 = *(_BYTE **)(v24 + 8);
    if ( *v5 == 45 && !v5[1] )
      goto LABEL_5;
    v6 = qword_4CFFD90;
    if ( !qword_4CFFD90 )
      goto LABEL_5;
    while ( v24 != *(_QWORD *)v6 )
    {
      v6 = *(_QWORD *)(v6 + 16);
      if ( !v6 )
        BUG();
    }
    v7 = *(__int16 *)(v6 + 24);
    if ( v7 <= 0 )
    {
LABEL_21:
      v9 = 1;
      qword_4CFDEB0 = (FILE *)sub_7214D0(v5, a2);
      if ( !qword_4CFDEB0 )
      {
LABEL_5:
        if ( !dword_4CFDEAC )
        {
          dword_4CFDEAC = 1;
          dword_4CFDEA8 = a1;
          dword_4CFDEA0 = 0;
        }
        return 0;
      }
    }
    else
    {
      v8 = 0;
      while ( 1 )
      {
        v9 = *(_DWORD *)(v6 + 4 * v8 + 28);
        v10 = (int)v8;
        if ( v22 < v9 )
          break;
        if ( v7 <= (int)++v8 )
          goto LABEL_32;
      }
      if ( !(_DWORD)v8 )
        goto LABEL_21;
      v10 = (int)v8 - 1;
      v9 = *(_DWORD *)(v6 + 4LL * (int)v10 + 28);
LABEL_32:
      v13 = *(_QWORD *)(v6 + 8 * v10 + 72);
      qword_4CFDEB0 = (FILE *)sub_7214D0(v5, a2);
      if ( !qword_4CFDEB0 )
        goto LABEL_5;
      if ( v13 && fseek(qword_4CFDEB0, v13, 0) )
      {
LABEL_35:
        fclose(qword_4CFDEB0);
        qword_4CFDEB0 = 0;
        goto LABEL_5;
      }
    }
    sub_722830(v25, (unsigned int)*a2);
    v11 = v22 - v9;
    if ( v22 != v9 )
    {
      do
      {
        while ( 1 )
        {
          v12 = v26 > 1 ? sub_722840(qword_4CFDEB0) : getc(qword_4CFDEB0);
          if ( v12 == 10 )
            break;
          if ( v12 == -1 )
            goto LABEL_35;
        }
        --v11;
      }
      while ( v11 );
    }
    v14 = (char *)qword_4CFFDA0;
    if ( !qword_4CFFDA0 )
    {
      qword_4CFFDA0 = sub_822BE0(201);
      v14 = (char *)qword_4CFFDA0;
      qword_4CFFD98 = qword_4CFFDA0 + 200;
    }
    v15 = qword_4CFFD98 - 2;
    while ( 1 )
    {
      v16 = v26 > 1 ? sub_722840(qword_4CFDEB0) : getc(qword_4CFDEB0);
      if ( v16 == -1 || v16 == 10 )
        break;
      if ( v14 == (char *)v15 )
      {
        v21 = v16;
        v17 = qword_4CFFD98 - qword_4CFFDA0;
        v18 = &v14[-qword_4CFFDA0];
        v19 = sub_822C60(qword_4CFFDA0, qword_4CFFD98 - qword_4CFFDA0 + 1, qword_4CFFD98 - qword_4CFFDA0 + 1001);
        v16 = v21;
        qword_4CFFDA0 = v19;
        v14 = &v18[v19];
        qword_4CFFD98 = v19 + v17 + 1000;
        v15 = v19 + v17 + 998;
      }
      if ( !v16 )
        LOBYTE(v16) = 32;
      *v14++ = v16;
    }
    *(_WORD *)v14 = 10;
    fclose(qword_4CFDEB0);
    qword_4CFDEB0 = 0;
    if ( !dword_4CFDEAC )
    {
      v20 = *a2;
      dword_4CFDEA8 = a1;
      dword_4CFDEAC = 1;
      dword_4CFDEA0 = 1;
      dword_4CFDEA4 = v20;
    }
    return 1;
  }
  return result;
}
