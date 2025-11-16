// Function: sub_7BE9B0
// Address: 0x7be9b0
//
_QWORD *__fastcall sub_7BE9B0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int16 v10; // ax
  _QWORD *result; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _WORD *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _WORD v25[176]; // [rsp+0h] [rbp-180h] BYREF
  int v26; // [rsp+160h] [rbp-20h]
  __int16 v27; // [rsp+164h] [rbp-1Ch]

  while ( word_4F06418[0] == 92 )
  {
    sub_7B8B50(a1, (unsigned int *)a2, a3, a4, a5, a6);
    v10 = word_4F06418[0];
    if ( word_4F06418[0] == 38 )
    {
      a2 = 0;
      a1 = 0;
      if ( (unsigned __int16)sub_7BE840(0, 0) == 245 )
        sub_7B8B50(0, 0, v6, v7, v8, v9);
      v10 = word_4F06418[0];
    }
    if ( v10 != 245 )
    {
      if ( v10 == 244 )
      {
        sub_7B8B50(a1, (unsigned int *)a2, v6, v7, v8, v9);
        v10 = word_4F06418[0];
      }
      if ( v10 != 27 )
      {
        a2 = (__int64)dword_4F07508;
        a1 = 125;
        sub_6851C0(0x7Du, dword_4F07508);
        result = (_QWORD *)sub_7BE9B0();
        if ( word_4F06418[0] != 86 )
          return result;
        goto LABEL_8;
      }
      a1 = 0;
      sub_7BDC20(0, a2, v6, v7, v8, v9);
    }
    sub_7B8B50(a1, (unsigned int *)a2, v6, v7, v8, v9);
    result = (_QWORD *)sub_7BE9B0();
    if ( word_4F06418[0] != 86 )
      return result;
LABEL_8:
    sub_7B8B50(a1, (unsigned int *)a2, v12, v13, v14, v15);
  }
  sub_7B8190();
  *((_BYTE *)qword_4F061C0 + 56) |= 8u;
  if ( word_4F06418[0] == 73 )
  {
    v20 = 0;
    sub_7BDC20(0, a2, v16, v17, v18, v19);
  }
  else
  {
    a2 = 1;
    memset(v25, 0, sizeof(v25));
    HIBYTE(v25[4]) = 1;
    v25[37] = 257;
    v26 = 0;
    v27 = 0;
    v20 = v25;
    sub_7BDFF0((unsigned __int64)v25, (unsigned int *)1, 257, 0, (__int64)v25, v19);
  }
  result = sub_7B8260();
  if ( word_4F06418[0] != 9 )
    return (_QWORD *)sub_7B8B50((unsigned __int64)v20, (unsigned int *)a2, v21, v22, v23, v24);
  return result;
}
