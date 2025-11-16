// Function: sub_1695A80
// Address: 0x1695a80
//
void __fastcall sub_1695A80(char *a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  unsigned __int64 v5; // r12
  __int64 v6; // r15
  size_t v7; // rdx
  _BYTE *v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // [rsp+10h] [rbp-50h]
  _QWORD v14[7]; // [rsp+28h] [rbp-38h] BYREF

  *a3 = 0;
  *a4 = 8;
  if ( a2 )
  {
    v5 = a2;
    v6 = a2;
    v7 = 0x7FFFFFFFFFFFFFFFLL;
    if ( a2 >= 0 )
      v7 = a2;
    v9 = memchr(a1, 58, v7);
    if ( !v9 || (v10 = v9 - a1, v10 == -1) )
    {
      if ( (unsigned __int8)sub_16D2BB0(a1, a2, 10, v14) )
        return;
      goto LABEL_15;
    }
    if ( v10 )
    {
      if ( v10 <= a2 )
        a2 = v10;
      v13 = v10;
      if ( !(unsigned __int8)sub_16D2BB0(a1, a2, 10, v14) )
        *a3 = v14[0];
      if ( v5 - 1 > v13 )
      {
        v11 = v13 + 1;
        v12 = 0;
        if ( v13 + 1 > v5 )
          goto LABEL_20;
        goto LABEL_19;
      }
    }
    else if ( a2 != 1 )
    {
      v11 = 1;
LABEL_19:
      v6 = v11;
      v12 = v5 - v11;
LABEL_20:
      if ( (unsigned __int8)sub_16D2BB0(&a1[v6], v12, 10, v14) )
        return;
LABEL_15:
      *a4 = v14[0];
    }
  }
}
