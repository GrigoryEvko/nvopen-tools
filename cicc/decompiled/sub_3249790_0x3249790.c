// Function: sub_3249790
// Address: 0x3249790
//
__int64 __fastcall sub_3249790(__int64 *a1, __int64 a2, __int16 a3, __int16 a4, __int64 **a5)
{
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  _BYTE *v11; // rsi
  unsigned __int64 **v12; // r12
  __int64 result; // rax
  unsigned int v14; // r15d
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17[8]; // [rsp+10h] [rbp-40h] BYREF

  v9 = a1[23];
  v16 = (__int64)a5;
  v10 = sub_31DF6E0(v9);
  LODWORD(v17[0]) = v10;
  WORD2(v17[0]) = WORD2(v10);
  sub_3215F30(a5, (__int64)v17);
  v11 = (_BYTE *)a1[34];
  if ( v11 == (_BYTE *)a1[35] )
  {
    sub_3248DF0((__int64)(a1 + 33), v11, &v16);
  }
  else
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = v16;
      v11 = (_BYTE *)a1[34];
    }
    a1[34] = (__int64)(v11 + 8);
  }
  v12 = (unsigned __int64 **)(a2 + 8);
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v14 = (unsigned __int16)sub_3220AA0(a1[26]), result = sub_E06A90(a3), v14 >= (unsigned int)result) )
  {
    LODWORD(v17[0]) = 8;
    HIWORD(v17[0]) = a4;
    WORD2(v17[0]) = a3;
    v17[1] = v16;
    return sub_3248F80(v12, a1 + 11, v17);
  }
  return result;
}
