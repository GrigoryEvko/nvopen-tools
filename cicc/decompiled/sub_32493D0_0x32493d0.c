// Function: sub_32493D0
// Address: 0x32493d0
//
__int64 __fastcall sub_32493D0(__int64 *a1, unsigned __int64 a2, __int16 a3, unsigned __int64 a4)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r13
  unsigned __int64 **v9; // r8
  __int16 v10; // r13
  __int64 result; // rax
  unsigned int v12; // r12d
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // [rsp+8h] [rbp-48h]
  unsigned __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16[8]; // [rsp+10h] [rbp-40h] BYREF

  v14 = sub_3215100(a2);
  v6 = sub_3215100(a4);
  v7 = v14;
  v8 = v6;
  if ( !v14 )
    v7 = sub_3215100((unsigned __int64)(a1 + 1));
  if ( !v8 )
  {
    v15 = v7;
    v13 = sub_3215100((unsigned __int64)(a1 + 1));
    v7 = v15;
    v8 = v13;
  }
  v9 = (unsigned __int64 **)(a2 + 8);
  v10 = 3 * (v7 == v8) + 16;
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v12 = (unsigned __int16)sub_3220AA0(a1[26]),
        result = sub_E06A90(a3),
        v9 = (unsigned __int64 **)(a2 + 8),
        v12 >= (unsigned int)result) )
  {
    WORD2(v16[0]) = a3;
    LODWORD(v16[0]) = 7;
    HIWORD(v16[0]) = v10;
    v16[1] = a4;
    return sub_3248F80(v9, a1 + 11, v16);
  }
  return result;
}
