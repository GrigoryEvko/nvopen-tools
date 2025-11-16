// Function: sub_67A1A0
// Address: 0x67a1a0
//
__int64 __fastcall sub_67A1A0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  char v4; // r14
  unsigned __int16 v5; // r12
  int v6; // eax
  __int64 v7; // rdx
  unsigned __int64 v8; // rdx
  unsigned int v9; // r15d
  __int64 v10; // rdx
  __int64 result; // rax
  unsigned int v12; // r13d
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  char v23; // al
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v34; // [rsp+10h] [rbp-50h]
  int v35; // [rsp+18h] [rbp-48h]
  unsigned int v36; // [rsp+1Ch] [rbp-44h]
  _WORD v37[25]; // [rsp+2Eh] [rbp-32h] BYREF

  v4 = 0;
  v5 = a2;
  v34 = a1;
  v36 = a2;
  v6 = word_4F06418[0];
  v35 = a4;
LABEL_2:
  while ( 2 )
  {
    v7 = (unsigned int)(v6 - 33);
    if ( (unsigned __int16)(v6 - 33) <= 1u )
      goto LABEL_12;
    while ( 1 )
    {
      a4 = dword_4D04474;
      if ( !dword_4D04474 || (_WORD)v6 != 52 )
        break;
      do
      {
LABEL_12:
        v9 = v5;
        a2 = 0;
        v4 = 1;
        a1 = v5;
        sub_679930(v5, 0, v7, a4);
        v6 = word_4F06418[0];
        if ( word_4F06418[0] != 25 || !dword_4D043F8 )
          goto LABEL_2;
LABEL_9:
        a2 = 0;
        a1 = 0;
        if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
        {
          a1 = v9;
          sub_679A30(v9, 0, v10, a4);
        }
        v6 = word_4F06418[0];
        v7 = (unsigned int)word_4F06418[0] - 33;
      }
      while ( (unsigned __int16)(word_4F06418[0] - 33) <= 1u );
    }
    v8 = (unsigned int)(v6 - 81);
    if ( (unsigned __int16)(v6 - 81) <= 0x26u && (a4 = 0x6004000001LL, _bittest64(&a4, v8))
      || (v8 = (unsigned int)(v6 - 263), (unsigned __int16)(v6 - 263) <= 3u)
      || (_WORD)v6 == 15 )
    {
      v9 = v5;
      a2 = 0;
      a1 = v5;
      sub_679930(v5, 0, v8, a4);
      v6 = word_4F06418[0];
      if ( word_4F06418[0] != 25 || !dword_4D043F8 )
        continue;
      goto LABEL_9;
    }
    break;
  }
  if ( (_WORD)v6 != 142 )
    goto LABEL_18;
  sub_7B8B50(a1, a2, v8, a4);
  LOWORD(v6) = word_4F06418[0];
  if ( word_4F06418[0] == 27 )
  {
    a1 = (unsigned __int16)v36;
    sub_679AE0((unsigned __int16)v36, a2, v8, a4);
    LOWORD(v6) = word_4F06418[0];
LABEL_18:
    if ( (_WORD)v6 == 27 )
    {
      v12 = (unsigned __int16)v36;
      v13 = (unsigned __int16)v36;
      sub_679930((unsigned __int16)v36, 0, v8, a4);
      v15 = word_4F06418[0];
      if ( !qword_4D0495C )
      {
        v16 = v36 & 1;
        goto LABEL_62;
      }
      if ( !v35 )
      {
        v16 = v36 & 1;
        goto LABEL_62;
      }
      if ( (v36 & 0x40) != 0 )
      {
        v16 = v36 & 1;
        goto LABEL_62;
      }
      if ( word_4F06418[0] == 1 )
      {
        v13 = 28;
        if ( (unsigned __int16)sub_7BEB10(28, v37) == 28 )
        {
          if ( v37[0] != 27 )
            goto LABEL_39;
          v15 = word_4F06418[0];
          v16 = v36 & 1;
          goto LABEL_62;
        }
        v15 = word_4F06418[0];
      }
      if ( (_WORD)v15 == 28 )
        goto LABEL_39;
      v14 = v36;
      v16 = v36 & 1;
      v23 = v36 & (v4 ^ 1);
      if ( v23 )
      {
        if ( (unsigned __int16)(v15 - 80) > 0x30u )
        {
          if ( (_WORD)v15 == 165 || (_WORD)v15 == 180 || (unsigned __int16)(v15 - 331) <= 4u || (_WORD)v15 == 18 )
            goto LABEL_39;
        }
        else
        {
          v24 = 0x1C70006066221LL;
          if ( _bittest64(&v24, (unsigned int)(v15 - 80)) )
            goto LABEL_39;
        }
        if ( unk_4D04548 | unk_4D04558 && (unsigned __int16)(v15 - 133) <= 3u
          || (_WORD)v15 == 239
          || (unsigned __int16)(v15 - 272) <= 8u
          || (_DWORD)qword_4F077B4 && ((_WORD)v15 == 236 || (unsigned __int16)(v15 - 339) <= 0xFu) )
        {
          goto LABEL_39;
        }
        if ( (unsigned __int16)(v15 - 151) > 0x27u )
        {
          if ( (unsigned __int16)(v15 - 87) <= 0x11u )
          {
            if ( ((0x24001uLL >> ((unsigned __int8)v15 - 87)) & 1) != 0 )
              goto LABEL_39;
            goto LABEL_63;
          }
        }
        else
        {
          v23 = ((0xC500000001uLL >> ((unsigned __int8)v15 + 105)) & 1) == 0;
        }
        if ( !v23 || (_WORD)v15 == 236 || dword_4F0775C && (_WORD)v15 == 77 || (_WORD)v15 == 1 )
          goto LABEL_39;
LABEL_63:
        if ( (_WORD)v15 == 28 )
          goto LABEL_30;
        v13 = 6;
        if ( (unsigned int)sub_651B00(6u) )
          goto LABEL_30;
        v15 = word_4F06418[0];
        if ( word_4F06418[0] == 76 )
        {
          v13 = 0;
          if ( (unsigned __int16)sub_7BE840(0, 0) == 28 )
            goto LABEL_30;
          v15 = word_4F06418[0];
        }
LABEL_66:
        if ( (_WORD)v15 == 142 )
        {
          sub_7B8B50(v13, v15, v16, v14);
          LOWORD(v15) = word_4F06418[0];
          if ( word_4F06418[0] == 27 )
          {
            sub_679AE0((unsigned __int16)v36, word_4F06418[0], v27, v28);
            LOWORD(v15) = word_4F06418[0];
          }
        }
        if ( (_WORD)v15 != 1 || (unk_4D04A12 & 1) == 0 || !HIDWORD(qword_4F077B4) || (v36 & 0x40) != 0 )
        {
          result = sub_67A1A0(v34, (unsigned __int16)v36, 0, 0);
          if ( !*(_DWORD *)(v34 + 12) || *(_DWORD *)(v34 + 16) )
            return result;
          if ( word_4F06418[0] == 28 )
          {
            a2 = 0;
            a1 = (unsigned __int16)v36;
            sub_679930((unsigned __int16)v36, 0, v17, v18);
            goto LABEL_22;
          }
        }
        goto LABEL_39;
      }
LABEL_62:
      if ( !(_WORD)v16 )
        goto LABEL_66;
      goto LABEL_63;
    }
  }
  if ( (_WORD)v6 != 76 )
    goto LABEL_20;
  if ( !dword_4D04408 )
    goto LABEL_21;
  a1 = (unsigned __int16)v36;
  a2 = 0;
  sub_679930((unsigned __int16)v36, 0, v8, a4);
LABEL_20:
  result = word_4F06418[0];
  if ( word_4F06418[0] != 1 )
  {
LABEL_21:
    if ( (v36 & 2) == 0 )
    {
LABEL_22:
      result = word_4F06418[0];
      if ( word_4F06418[0] == 25 )
      {
        v8 = (unsigned __int64)&dword_4D043F8;
        if ( dword_4D043F8 )
        {
          a2 = 0;
          a1 = 0;
          if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
          {
            a1 = (unsigned __int16)v36;
            sub_679A30((unsigned __int16)v36, 0, v8, a4);
          }
          result = word_4F06418[0];
        }
      }
      while ( 1 )
      {
LABEL_27:
        while ( (_WORD)result == 27 )
        {
          v12 = (unsigned __int16)v36;
          sub_679930((unsigned __int16)v36, 0, v8, a4);
          if ( word_4F06418[0] != 76 && word_4F06418[0] != 28 && (a3 & 1) != 0 && !(unsigned int)sub_679C10(3u) )
          {
            v19 = 1;
            v20 = 28;
            sub_6797D0(0x1Cu, 1);
            result = word_4F06418[0];
            if ( word_4F06418[0] == 28 )
            {
              v19 = 0;
              v20 = (unsigned __int16)v36;
              sub_679930((unsigned __int16)v36, 0, v21, v22);
              result = word_4F06418[0];
            }
            if ( (_WORD)result == 142 )
            {
              result = sub_7B8B50(v20, v19, v21, v22);
              if ( word_4F06418[0] == 27 )
                result = (__int64)sub_679AE0((unsigned __int16)v36, v19, v31, v32);
            }
            if ( !v35 )
              return result;
            goto LABEL_50;
          }
LABEL_30:
          a1 = v34;
          a2 = v12;
          sub_67B070(v34, v12);
          result = word_4F06418[0];
        }
        if ( (_WORD)result != 25 )
          break;
        sub_679930(v5, 0, v8, a4);
        a2 = 1;
        a1 = 26;
        sub_6797D0(0x1Au, 1);
        result = word_4F06418[0];
        if ( word_4F06418[0] == 26 )
        {
          a2 = 0;
          a1 = v5;
          sub_679930(v5, 0, v8, a4);
          result = word_4F06418[0];
        }
      }
      if ( (_WORD)result == 142 )
      {
        sub_7B8B50(a1, a2, v8, a4);
        result = word_4F06418[0];
        if ( word_4F06418[0] == 27 )
        {
          sub_679AE0((unsigned __int16)v36, a2, v25, v26);
          result = word_4F06418[0];
        }
      }
      if ( (_WORD)result == 56 )
      {
        if ( !v35 || (v36 & 0xFFFC) != 0 )
          return sub_679830();
        *(_DWORD *)(v34 + 16) = 1;
        return v34;
      }
      if ( !v35 )
        return result;
      if ( (_WORD)result == 73 )
      {
        result = (__int64)&dword_4D04428;
        if ( dword_4D04428 )
        {
          if ( (v36 & 2) != 0 )
          {
            result = sub_7C6040(0, 0);
            if ( word_4F06418[0] == 74 )
              return sub_7B8B50(0, 0, v29, v30);
            return result;
          }
        }
      }
LABEL_50:
      if ( (v36 & 8) != 0 )
        goto LABEL_39;
      return result;
    }
LABEL_38:
    if ( (v36 & 1) != 0 )
      goto LABEL_22;
LABEL_39:
    result = v34;
    *(_DWORD *)(v34 + 12) = 0;
    return result;
  }
  a4 = unk_4D04A10;
  if ( (unk_4D04A10 & 2) != 0 && (v36 & 0x40) == 0
    || (a1 = v34, v8 = v36 & 2, *(_DWORD *)(v34 + 8))
    && (a2 = (__int64)&dword_4F077BC, dword_4F077BC)
    && !v35
    && (a4 = unk_4D04A10 & 1, (unk_4D04A10 & 1) != 0) )
  {
    if ( (v36 & 2) == 0 )
      goto LABEL_27;
    goto LABEL_38;
  }
  if ( (v36 & 2) == 0 )
    goto LABEL_27;
  a1 = (unsigned __int16)v36;
  a2 = 0;
  sub_679930((unsigned __int16)v36, 0, v8, a4);
  if ( !*(_DWORD *)(v34 + 20) )
    goto LABEL_22;
  result = 0;
  if ( (unk_4D04A12 & 2) != 0 )
    result = xmmword_4D04A20.m128i_i64[0];
  *(_QWORD *)v34 = result;
  *(_DWORD *)(v34 + 12) = 0;
  return result;
}
