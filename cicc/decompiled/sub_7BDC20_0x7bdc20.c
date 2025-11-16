// Function: sub_7BDC20
// Address: 0x7bdc20
//
__int64 __fastcall sub_7BDC20(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  _QWORD *v7; // rdx
  char v8; // si
  _BOOL8 v9; // rsi
  __int16 v10; // bx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int *v15; // rsi
  unsigned int v16; // r15d
  unsigned __int64 v17; // rdi
  _BOOL4 v18; // r14d
  __int64 v19; // rax
  __int64 result; // rax
  bool v21; // [rsp+13h] [rbp-5Dh]
  unsigned int v22; // [rsp+14h] [rbp-5Ch]
  unsigned __int64 v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h]
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h]
  __int64 v27; // [rsp+38h] [rbp-38h]

  v6 = a1;
  v7 = qword_4F061C0;
  v8 = *((_BYTE *)qword_4F061C0 + 56);
  *((_BYTE *)qword_4F061C0 + 56) = v8 | 8;
  v9 = (v8 & 8) != 0;
  v21 = v9;
  v22 = dword_4F063F8;
  if ( word_4F06418[0] == 43 )
  {
    v23 = 2;
    v10 = 44;
    goto LABEL_6;
  }
  if ( word_4F06418[0] > 0x2Bu )
  {
    if ( word_4F06418[0] == 73 )
    {
      v23 = 20;
      v10 = 74;
      goto LABEL_6;
    }
LABEL_52:
    sub_721090();
  }
  if ( word_4F06418[0] == 25 )
  {
    v23 = 2;
    v10 = 26;
    goto LABEL_6;
  }
  if ( word_4F06418[0] != 27 )
    goto LABEL_52;
  v23 = 2;
  v10 = 28;
LABEL_6:
  sub_7B8B50(a1, (unsigned int *)v9, (__int64)v7, a4, a5, a6);
  v27 = 0;
  v15 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  while ( 1 )
  {
    v16 = word_4F06418[0];
    if ( word_4F06418[0] == v10 && !(v24 | v25 | v26) )
      break;
    switch ( word_4F06418[0] )
    {
      case 0x19u:
        ++v25;
        v17 = word_4F06418[0];
        v18 = 0;
        goto LABEL_22;
      case 0x1Au:
        if ( !v25 )
          goto LABEL_38;
        v17 = word_4F06418[0];
        v18 = 0;
        --v25;
        goto LABEL_22;
      case 0x1Bu:
        ++v26;
        v17 = word_4F06418[0];
        v18 = 0;
        goto LABEL_22;
      case 0x1Cu:
        if ( !v26 )
          goto LABEL_38;
        v17 = word_4F06418[0];
        v18 = 0;
        --v26;
        goto LABEL_22;
      case 0x2Au:
        v12 = dword_4F07770;
        if ( !dword_4F07770 || v10 != 44 )
          goto LABEL_38;
        v18 = 0;
        v17 = word_4F06418[0];
        if ( v24 | v25 | v26 )
          goto LABEL_22;
        sub_7BC010(word_4F06418[0]);
        v24 = 0;
        v25 = 0;
        v26 = 0;
        v16 = (unsigned int)v15;
        goto LABEL_19;
      case 0x2Cu:
        v17 = word_4F06418[0];
        v18 = 0;
        if ( v10 == 44 )
        {
          v11 = v26;
          v14 = v25;
          v17 = 44;
          v12 = v24;
          v19 = v24 | v25 | v26;
          v18 = v19 == 0;
          v13 = 0;
          if ( v19 )
          {
            v13 = v26;
          }
          else
          {
            v12 = 0;
            v14 = 0;
          }
          v24 = v12;
          v25 = v14;
          v26 = v13;
        }
        goto LABEL_22;
      case 0x49u:
        ++v24;
        v17 = word_4F06418[0];
        v18 = 0;
        goto LABEL_22;
      case 0x4Au:
        if ( v24 )
        {
          v17 = word_4F06418[0];
          v18 = 0;
          --v24;
        }
        else
        {
LABEL_38:
          v17 = word_4F06418[0];
          v18 = 0;
        }
LABEL_22:
        if ( !v6 )
          goto LABEL_15;
        goto LABEL_10;
      default:
        v17 = word_4F06418[0];
        v18 = 0;
        if ( !v6 )
          goto LABEL_11;
LABEL_10:
        if ( dword_4F063F8 - v22 > v23 )
          goto LABEL_40;
LABEL_11:
        if ( (_WORD)v17 == 43 && v10 == 44 && (_WORD)v15 == 1 && dword_4F077C4 == 2 )
        {
          v11 = unk_4D03D20;
          if ( !unk_4D03D20 )
          {
            if ( v27 )
            {
              v17 = v27;
              if ( (unsigned int)sub_7AD1A0(v27) )
                sub_7BDC10();
              v16 = word_4F06418[0];
            }
          }
        }
LABEL_15:
        if ( (_WORD)v16 == 9 || dword_4D03D18 && (_WORD)v16 == 10 )
          goto LABEL_40;
        v27 = qword_4D04A00;
        sub_7B8B50(v17, v15, v11, v12, v13, v14);
        if ( v18 )
          goto LABEL_40;
LABEL_19:
        v15 = (unsigned int *)v16;
        break;
    }
  }
LABEL_40:
  result = (8 * v21) | (_BYTE)qword_4F061C0[7] & 0xF7u;
  *((_BYTE *)qword_4F061C0 + 56) = (8 * v21) | qword_4F061C0[7] & 0xF7;
  return result;
}
