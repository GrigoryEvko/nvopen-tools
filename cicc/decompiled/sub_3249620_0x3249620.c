// Function: sub_3249620
// Address: 0x3249620
//
__int64 __fastcall sub_3249620(__int64 *a1, __int64 a2, __int16 a3, __int64 **a4)
{
  __int64 v7; // rdi
  unsigned __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // r12
  __int16 v11; // r15
  int v12; // eax
  unsigned __int64 **v13; // r8
  __int64 result; // rax
  unsigned int v15; // r12d
  __int64 v17; // [rsp+8h] [rbp-48h] BYREF
  __int64 v18[8]; // [rsp+10h] [rbp-40h] BYREF

  v7 = a1[23];
  v17 = (__int64)a4;
  v8 = sub_31DF6E0(v7);
  LODWORD(v18[0]) = v8;
  WORD2(v18[0]) = WORD2(v8);
  sub_3215EC0(a4, (__int64)v18);
  v9 = (_BYTE *)a1[37];
  if ( v9 == (_BYTE *)a1[38] )
  {
    sub_3248C60((__int64)(a1 + 36), v9, &v17);
    v10 = v17;
  }
  else
  {
    v10 = v17;
    if ( v9 )
    {
      *(_QWORD *)v9 = v17;
      v9 = (_BYTE *)a1[37];
      v10 = v17;
    }
    a1[37] = (__int64)(v9 + 8);
  }
  v11 = 24;
  if ( (unsigned __int16)sub_3220AA0(a1[26]) <= 3u )
  {
    v12 = *(_DWORD *)(v10 + 8);
    v11 = 10;
    if ( (v12 & 0xFFFFFF00) != 0 )
      v11 = ((v12 & 0xFFFF0000) != 0) + 3;
  }
  v13 = (unsigned __int64 **)(a2 + 8);
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v15 = (unsigned __int16)sub_3220AA0(a1[26]),
        result = sub_E06A90(a3),
        v13 = (unsigned __int64 **)(a2 + 8),
        v15 >= (unsigned int)result) )
  {
    LODWORD(v18[0]) = 9;
    WORD2(v18[0]) = a3;
    HIWORD(v18[0]) = v11;
    v18[1] = v17;
    return sub_3248F80(v13, a1 + 11, v18);
  }
  return result;
}
