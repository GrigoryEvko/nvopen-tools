// Function: sub_1665DD0
// Address: 0x1665dd0
//
void __fastcall sub_1665DD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  const char *v4; // rax
  __int64 v5; // r14
  _BYTE *v6; // rax
  __int64 v7; // rax
  char v8; // al
  char v9; // al
  char v10; // al
  char v11; // al
  const char *v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-40h] BYREF
  char v14; // [rsp+10h] [rbp-30h]
  char v15; // [rsp+11h] [rbp-2Fh]

  v2 = **(_QWORD **)(a2 - 48);
  if ( v2 == **(_QWORD **)(a2 - 24) )
  {
    v3 = *(_QWORD *)a2;
    switch ( *(_BYTE *)(a2 + 16) )
    {
      case '#':
      case '%':
      case '\'':
      case ')':
      case '*':
      case ',':
      case '-':
        v8 = *(_BYTE *)(v3 + 8);
        if ( v8 == 16 )
          v8 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
        if ( v8 != 11 )
        {
          v15 = 1;
          v12 = "Integer arithmetic operators only work with integral types!";
          goto LABEL_33;
        }
        if ( v3 == v2 )
          goto LABEL_15;
        v15 = 1;
        v4 = "Integer arithmetic operators must have same type for operands and result!";
        goto LABEL_4;
      case '$':
      case '&':
      case '(':
      case '+':
      case '.':
        v9 = *(_BYTE *)(v3 + 8);
        if ( v9 == 16 )
          v9 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
        if ( (unsigned __int8)(v9 - 1) > 5u )
        {
          v15 = 1;
          v4 = "Floating-point arithmetic operators only work with floating-point types!";
        }
        else
        {
          if ( v2 == v3 )
            goto LABEL_15;
          v15 = 1;
          v4 = "Floating-point arithmetic operators must have same type for operands and result!";
        }
        goto LABEL_4;
      case '/':
      case '0':
      case '1':
        v11 = *(_BYTE *)(v3 + 8);
        if ( v11 == 16 )
          v11 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
        if ( v11 != 11 )
        {
          v15 = 1;
          v12 = "Shifts only work with integral types!";
          goto LABEL_33;
        }
        if ( v2 == v3 )
          goto LABEL_15;
        v15 = 1;
        v4 = "Shift return type must be same as operands!";
        goto LABEL_4;
      case '2':
      case '3':
      case '4':
        v10 = *(_BYTE *)(v3 + 8);
        if ( v10 == 16 )
          v10 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
        if ( v10 == 11 )
        {
          if ( v2 == v3 )
          {
LABEL_15:
            sub_1663F80(a1, a2);
            return;
          }
          v15 = 1;
          v4 = "Logical operators must have same type for operands and result!";
          goto LABEL_4;
        }
        v15 = 1;
        v12 = "Logical operators only work with integral types!";
LABEL_33:
        v13[0] = v12;
        v14 = 3;
        sub_164FF40((__int64 *)a1, (__int64)v13);
        if ( *(_QWORD *)a1 )
          goto LABEL_8;
        break;
    }
  }
  else
  {
    v15 = 1;
    v4 = "Both operands to a binary operator are not of the same type!";
LABEL_4:
    v5 = *(_QWORD *)a1;
    v13[0] = v4;
    v14 = 3;
    if ( v5 )
    {
      sub_16E2CE0(v13, v5);
      v6 = *(_BYTE **)(v5 + 24);
      if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 16) )
      {
        sub_16E7DE0(v5, 10);
      }
      else
      {
        *(_QWORD *)(v5 + 24) = v6 + 1;
        *v6 = 10;
      }
      v7 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 72) = 1;
      if ( v7 )
LABEL_8:
        sub_164FA80((__int64 *)a1, a2);
    }
    else
    {
      *(_BYTE *)(a1 + 72) = 1;
    }
  }
}
