// Function: sub_67BBF0
// Address: 0x67bbf0
//
__int64 __fastcall sub_67BBF0(unsigned __int8 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax

  v1 = a1;
  if ( a1 == 1 || (result = 16LL * a1 + 82867168, qword_4F073E0[2 * a1]) )
  {
    v4 = (_QWORD *)qword_4D039D8;
    v5 = *(_QWORD *)(qword_4D039D8 + 16);
    if ( (unsigned __int64)(v5 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
    {
      sub_823810();
      v4 = (_QWORD *)qword_4D039D8;
      v5 = *(_QWORD *)(qword_4D039D8 + 16);
    }
    *(_BYTE *)(v4[4] + v5) = 27;
    v6 = v4[2];
    v7 = v6 + 1;
    v4[2] = v6 + 1;
    if ( (unsigned __int64)(v6 + 2) > v4[1] )
    {
      sub_823810();
      v4 = (_QWORD *)qword_4D039D8;
      v7 = *(_QWORD *)(qword_4D039D8 + 16);
    }
    *(_BYTE *)(v4[4] + v7) = 91;
    v8 = v4[2];
    v9 = v8 + 1;
    v4[2] = v8 + 1;
    if ( a1 == 1 )
    {
      if ( (unsigned __int64)(v8 + 2) > v4[1] )
      {
        sub_823810();
        v4 = (_QWORD *)qword_4D039D8;
        v9 = *(_QWORD *)(qword_4D039D8 + 16);
      }
      *(_BYTE *)(v4[4] + v9) = 48;
      result = v4[2] + 1LL;
      v4[2] = result;
      if ( (unsigned __int64)(result + 1) <= v4[1] )
        goto LABEL_11;
    }
    else
    {
      sub_8238B0(v4, qword_4F073E0[2 * v1], qword_4F073E0[2 * v1 + 1]);
      v4 = (_QWORD *)qword_4D039D8;
      result = *(_QWORD *)(qword_4D039D8 + 16);
      if ( (unsigned __int64)(result + 1) <= *(_QWORD *)(qword_4D039D8 + 8) )
      {
LABEL_11:
        *(_BYTE *)(v4[4] + result) = 109;
        ++v4[2];
        return result;
      }
    }
    sub_823810();
    v4 = (_QWORD *)qword_4D039D8;
    result = *(_QWORD *)(qword_4D039D8 + 16);
    goto LABEL_11;
  }
  return result;
}
