// Function: sub_3249A20
// Address: 0x3249a20
//
__int64 __fastcall sub_3249A20(__int64 *a1, unsigned __int64 **a2, __int16 a3, int a4, __int64 a5)
{
  __int16 v6; // r12
  __int64 result; // rax
  unsigned __int16 v8; // r15
  unsigned int v9; // eax
  unsigned int v10; // r9d
  __int64 v11; // [rsp+0h] [rbp-50h]
  __int64 v12[8]; // [rsp+10h] [rbp-40h] BYREF

  v6 = a4;
  if ( (a4 & 0xFF0000) == 0 )
  {
    v6 = 11;
    if ( (a5 & 0xFFFFFFFFFFFFFF00LL) != 0 )
    {
      v6 = 5;
      if ( (a5 & 0xFFFFFFFFFFFF0000LL) != 0 )
        v6 = (a5 != (unsigned int)a5) + 6;
    }
  }
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v11 = a5, v8 = sub_3220AA0(a1[26]), v9 = sub_E06A90(a3), a5 = v11, v10 = v9, result = v8, v8 >= v10) )
  {
    WORD2(v12[0]) = a3;
    LODWORD(v12[0]) = 1;
    HIWORD(v12[0]) = v6;
    v12[1] = a5;
    return sub_3248F80(a2, a1 + 11, v12);
  }
  return result;
}
