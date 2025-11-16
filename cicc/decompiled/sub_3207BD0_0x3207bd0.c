// Function: sub_3207BD0
// Address: 0x3207bd0
//
__int64 __fastcall sub_3207BD0(__int64 a1, unsigned __int8 *a2)
{
  int v3; // r13d
  int v4; // r12d
  int v5; // r15d
  int v6; // eax
  __int16 v7; // ax
  unsigned __int8 v8; // al
  __int64 v9; // rsi
  unsigned __int16 v10; // ax
  __int64 result; // rax
  __int64 v12; // rax
  __int16 v13; // [rsp+18h] [rbp-38h] BYREF
  int v14; // [rsp+1Ah] [rbp-36h]
  __int16 v15; // [rsp+1Eh] [rbp-32h]

  v3 = 0;
  v4 = 0;
LABEL_2:
  v5 = 1;
  while ( 1 )
  {
    v6 = v5 ^ 1;
    LOBYTE(v6) = (a2 == 0) | v5 ^ 1;
    v5 = v6;
    if ( (_BYTE)v6 )
      break;
    v7 = sub_AF18C0((__int64)a2);
    switch ( v7 )
    {
      case '5':
        v3 |= 2u;
        v4 |= 0x200u;
LABEL_8:
        v8 = *(a2 - 16);
        if ( (v8 & 2) != 0 )
          goto LABEL_9;
LABEL_12:
        v9 = (__int64)&a2[-8 * ((v8 >> 2) & 0xF) - 16];
LABEL_10:
        a2 = *(unsigned __int8 **)(v9 + 24);
        goto LABEL_2;
      case '7':
        v8 = *(a2 - 16);
        v4 |= 0x1000u;
        if ( (v8 & 2) == 0 )
          goto LABEL_12;
LABEL_9:
        v9 = *((_QWORD *)a2 - 4);
        goto LABEL_10;
      case '&':
        v3 |= 1u;
        v4 |= 0x400u;
        goto LABEL_8;
    }
  }
  if ( !a2 )
    goto LABEL_18;
  v10 = sub_AF18C0((__int64)a2);
  if ( v10 == 31 )
    return sub_3207960(a1, (__int64)a2, v4);
  if ( v10 > 0x1Fu )
  {
    if ( v10 != 66 )
      goto LABEL_18;
    return sub_3206C30(a1, (__int64)a2, v4);
  }
  if ( (unsigned __int16)(v10 - 15) <= 1u )
    return sub_3206C30(a1, (__int64)a2, v4);
LABEL_18:
  result = sub_3206530(a1, a2, 0);
  if ( (_WORD)v3 )
  {
    v14 = result;
    v15 = v3;
    v13 = 4097;
    v12 = sub_3709240(a1 + 648, &v13);
    return sub_3707F80(a1 + 632, v12);
  }
  return result;
}
