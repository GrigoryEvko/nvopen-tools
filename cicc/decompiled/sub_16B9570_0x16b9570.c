// Function: sub_16B9570
// Address: 0x16b9570
//
__int64 __fastcall sub_16B9570(__int64 a1, _QWORD *a2, const char *a3, size_t a4, __int64 a5, __int64 a6, char a7)
{
  const char *v10; // r12
  __int64 v11; // rbx
  unsigned __int64 v12; // rsi
  _QWORD *v13; // rax
  _DWORD *v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  _DWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // r8
  const char *v25; // rax
  unsigned __int64 v27; // rsi
  _QWORD *v28; // rax
  _DWORD *v29; // r8
  _QWORD *v30; // rax
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rsi
  _DWORD *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned int v42; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v43; // [rsp+18h] [rbp-58h] BYREF
  int *v44[2]; // [rsp+20h] [rbp-50h] BYREF
  char v45; // [rsp+30h] [rbp-40h]
  char v46; // [rsp+31h] [rbp-3Fh]

  v10 = a3;
  v11 = a1;
  v42 = (unsigned int)a2;
  if ( !a7 )
  {
    v43 = sub_16D5D50(a1, a2, a3);
    v30 = *(_QWORD **)&dword_4FA0208[2];
    a2 = dword_4FA0208;
    if ( !*(_QWORD *)&dword_4FA0208[2] )
      goto LABEL_57;
    do
    {
      while ( 1 )
      {
        v31 = v30[2];
        v32 = v30[3];
        if ( v43 <= v30[4] )
          break;
        v30 = (_QWORD *)v30[3];
        if ( !v32 )
          goto LABEL_36;
      }
      a2 = v30;
      v30 = (_QWORD *)v30[2];
    }
    while ( v31 );
LABEL_36:
    if ( a2 == (_QWORD *)dword_4FA0208 || v43 < a2[4] )
    {
LABEL_57:
      v44[0] = (int *)&v43;
      a2 = (_QWORD *)sub_16B93F0(&qword_4FA0200, a2, (unsigned __int64 **)v44);
    }
    v33 = a2[7];
    if ( v33 )
    {
      a1 = *(unsigned int *)(a1 + 8);
      v34 = (__int64)(a2 + 6);
      do
      {
        while ( 1 )
        {
          v35 = *(_QWORD *)(v33 + 16);
          a3 = *(const char **)(v33 + 24);
          if ( *(_DWORD *)(v33 + 32) >= (int)a1 )
            break;
          v33 = *(_QWORD *)(v33 + 24);
          if ( !a3 )
            goto LABEL_43;
        }
        v34 = v33;
        v33 = *(_QWORD *)(v33 + 16);
      }
      while ( v35 );
LABEL_43:
      if ( a2 + 6 != (_QWORD *)v34 && (int)a1 >= *(_DWORD *)(v34 + 32) )
        goto LABEL_46;
    }
    else
    {
      v34 = (__int64)(a2 + 6);
    }
    a1 = (__int64)(a2 + 5);
    a2 = (_QWORD *)v34;
    v44[0] = (int *)(v11 + 8);
    v34 = sub_16B94C0((_QWORD *)a1, v34, v44);
LABEL_46:
    ++*(_DWORD *)(v34 + 36);
  }
  if ( (*(_BYTE *)(v11 + 12) & 7) != 0 )
  {
    if ( (*(_BYTE *)(v11 + 12) & 7) == 2 )
    {
      v12 = sub_16D5D50(a1, a2, a3);
      v13 = *(_QWORD **)&dword_4FA0208[2];
      if ( *(_QWORD *)&dword_4FA0208[2] )
      {
        v14 = dword_4FA0208;
        do
        {
          while ( 1 )
          {
            v15 = v13[2];
            v16 = v13[3];
            if ( v12 <= v13[4] )
              break;
            v13 = (_QWORD *)v13[3];
            if ( !v16 )
              goto LABEL_9;
          }
          v14 = v13;
          v13 = (_QWORD *)v13[2];
        }
        while ( v15 );
LABEL_9:
        if ( v14 != dword_4FA0208 && v12 >= *((_QWORD *)v14 + 4) )
        {
          v17 = *((_QWORD *)v14 + 7);
          if ( v17 )
          {
            v18 = *(unsigned int *)(v11 + 8);
            v19 = v14 + 12;
            do
            {
              while ( 1 )
              {
                v20 = *(_QWORD *)(v17 + 16);
                v21 = *(_QWORD *)(v17 + 24);
                if ( *(_DWORD *)(v17 + 32) >= (int)v18 )
                  break;
                v17 = *(_QWORD *)(v17 + 24);
                if ( !v21 )
                  goto LABEL_16;
              }
              v19 = (_DWORD *)v17;
              v17 = *(_QWORD *)(v17 + 16);
            }
            while ( v20 );
LABEL_16:
            if ( v19 != v14 + 12 && (int)v18 >= v19[8] && (int)v19[9] > 1 )
            {
              v22 = sub_16E8CB0(v19, v18, v21);
              v46 = 1;
              v24 = v22;
              v25 = "must occur exactly one time!";
LABEL_56:
              v44[0] = (int *)v25;
              v45 = 3;
              return sub_16B1F90(v11, (__int64)v44, v10, a4, v24, v23);
            }
          }
        }
      }
    }
  }
  else
  {
    v27 = sub_16D5D50(a1, a2, a3);
    v28 = *(_QWORD **)&dword_4FA0208[2];
    v29 = dword_4FA0208;
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      do
      {
        if ( v27 > v28[4] )
        {
          v28 = (_QWORD *)v28[3];
        }
        else
        {
          v29 = v28;
          v28 = (_QWORD *)v28[2];
        }
      }
      while ( v28 );
      if ( v29 != dword_4FA0208 && v27 >= *((_QWORD *)v29 + 4) )
      {
        v36 = *((_QWORD *)v29 + 7);
        if ( v36 )
        {
          v37 = *(unsigned int *)(v11 + 8);
          v38 = v29 + 12;
          do
          {
            while ( 1 )
            {
              v39 = *(_QWORD *)(v36 + 16);
              v40 = *(_QWORD *)(v36 + 24);
              if ( *(_DWORD *)(v36 + 32) >= (int)v37 )
                break;
              v36 = *(_QWORD *)(v36 + 24);
              if ( !v40 )
                goto LABEL_52;
            }
            v38 = (_DWORD *)v36;
            v36 = *(_QWORD *)(v36 + 16);
          }
          while ( v39 );
LABEL_52:
          if ( v29 + 12 != v38 && (int)v37 >= v38[8] && (int)v38[9] > 1 )
          {
            v41 = sub_16E8CB0(v38, v37, v40);
            v46 = 1;
            v24 = v41;
            v25 = "may only occur zero or one times!";
            goto LABEL_56;
          }
        }
      }
    }
  }
  if ( qword_4FA0160 )
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, const char *, size_t, __int64, __int64))(*(_QWORD *)qword_4FA0160 + 16LL))(
      qword_4FA0160,
      v11,
      *(unsigned int *)(v11 + 16),
      v42,
      v10,
      a4,
      a5,
      a6);
  return (**(__int64 (__fastcall ***)(__int64, _QWORD, const char *, size_t, __int64, __int64))v11)(
           v11,
           v42,
           v10,
           a4,
           a5,
           a6);
}
