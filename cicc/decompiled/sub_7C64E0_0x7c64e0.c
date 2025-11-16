// Function: sub_7C64E0
// Address: 0x7c64e0
//
_QWORD *__fastcall sub_7C64E0(unsigned __int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r15
  int v8; // r13d
  int v9; // r12d
  __int64 v10; // rsi
  int v11; // eax
  int v12; // edx
  char v13; // r12
  __int64 v14; // rdx
  unsigned __int16 v15; // ax
  __int64 *v16; // rdx
  int v17; // eax
  _QWORD *result; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rdx
  int v22; // [rsp+8h] [rbp-48h]
  int v23; // [rsp+Ch] [rbp-44h]
  unsigned int v24; // [rsp+10h] [rbp-40h]
  int v26; // [rsp+18h] [rbp-38h]
  char v27; // [rsp+1Dh] [rbp-33h]
  bool v28; // [rsp+1Eh] [rbp-32h]
  unsigned __int8 v29; // [rsp+1Fh] [rbp-31h]

  v6 = a1;
  v8 = a3 & 1;
  v27 = a3 & 1;
  v9 = a3 & 4;
  v28 = a1 != 0;
  v10 = a1 != 0;
  v22 = dword_4F06650[0];
  v23 = dword_4F04D80;
  dword_4F04D80 = 1;
  v29 = v10 & !(a3 & 1);
  if ( (a3 & 1) != 0 )
  {
    a1 = 1;
    sub_7BDB60(1);
    v11 = word_4F06418[0];
    if ( dword_4F077C4 == 2 )
    {
      if ( word_4F06418[0] != 1 || (v21 = &qword_4D04A00, (word_4D04A10 & 0x200) == 0) )
      {
        v10 = 0;
        a1 = 1;
        sub_7C0F00(1u, 0, (__int64)v21, a4, a5, a6);
        v11 = word_4F06418[0];
      }
    }
  }
  else
  {
    v11 = word_4F06418[0];
  }
  if ( !*(_BYTE *)(a2 + (unsigned __int16)v11) )
  {
    LOBYTE(v26) = 0;
    v12 = -(v9 == 0);
    v13 = 0;
    v14 = ((v12 & 0xFFFFC800) + 0x4000) | 1;
    v24 = v14;
    do
    {
      if ( v8
        && ((*(_BYTE *)(a2 + 44) || v13)
         && ((v14 = (unsigned int)(v11 - 176),
              LOBYTE(v14) = (unsigned __int16)(v11 - 176) <= 1u,
              LOBYTE(a4) = v14 | ((unsigned __int16)(v11 - 166) <= 1u),
              (_BYTE)a4)
          || v13)
         || (_WORD)v11 == 1
         && ((a1 = (unsigned __int64)&qword_4D04A00, (v19 = qword_4D04A18) != 0)
          || (v10 = 4, (v19 = (_QWORD *)sub_7D5DD0(&qword_4D04A00, 4)) != 0))
         && *((_BYTE *)v19 + 80) == 22) )
      {
        LOBYTE(v26) = 1;
        v15 = word_4F06418[0];
      }
      else
      {
        v17 = word_4F06418[0];
        if ( word_4F06418[0] == 160 )
        {
          v13 = 1;
          goto LABEL_12;
        }
        if ( word_4F06418[0] != 25 )
          goto LABEL_20;
        if ( !dword_4D043F8 )
          goto LABEL_32;
        v10 = 0;
        a1 = 0;
        if ( (unsigned __int16)sub_7BE840(0, 0) != 25 )
        {
          v17 = word_4F06418[0];
LABEL_20:
          v14 = v17 & 0xFFFFFFFD;
          if ( (v17 & 0xFFFD) != 0x19 && (_WORD)v17 != 73 && ((_WORD)v17 != 43 || (v26 & 1) == 0) )
          {
            LOBYTE(v26) = 0;
            if ( (_WORD)v17 == 9 )
              break;
            goto LABEL_11;
          }
LABEL_32:
          v10 = a3;
          a1 = v6;
          v26 = sub_7C6040(v6, a3, v14, a4, a5, a6);
          if ( v26 )
            break;
          v15 = word_4F06418[0];
          goto LABEL_10;
        }
        v10 = v29;
        a1 = v6;
        sub_7BBD70(v6, (unsigned int *)v29, v20, a4, a5, a6);
        LOBYTE(v26) = 0;
        v15 = word_4F06418[0];
      }
LABEL_10:
      if ( v15 == 9 )
        break;
LABEL_11:
      v13 = 0;
LABEL_12:
      if ( v29 )
      {
        a1 = v6;
        sub_7AE360(v6);
      }
      if ( v8 )
      {
        sub_7B8B50(a1, (unsigned int *)v10, v14, a4, a5, a6);
        v11 = word_4F06418[0];
        if ( dword_4F077C4 == 2 )
        {
          if ( word_4F06418[0] != 1 || (v16 = &qword_4D04A00, (word_4D04A10 & 0x200) == 0) )
          {
            a1 = v24;
            v10 = 0;
            sub_7C0F00(v24, 0, (__int64)v16, a4, a5, a6);
            v11 = word_4F06418[0];
          }
        }
      }
      else
      {
        sub_7B8B50(a1, (unsigned int *)v10, v14, a4, a5, a6);
        v11 = word_4F06418[0];
      }
      v14 = (unsigned __int16)v11;
    }
    while ( !*(_BYTE *)(a2 + (unsigned __int16)v11) );
  }
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  if ( v27 && v28 )
    sub_7AE700((__int64)(qword_4F061C0 + 3), v22, dword_4F06650[0], word_4F06418[0] == 9, v6);
  result = &dword_4F04D80;
  dword_4F04D80 = v23;
  if ( v8 )
    return sub_7BDC00();
  return result;
}
