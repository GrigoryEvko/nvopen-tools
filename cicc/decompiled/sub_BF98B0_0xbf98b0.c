// Function: sub_BF98B0
// Address: 0xbf98b0
//
void __fastcall sub_BF98B0(_BYTE *a1, unsigned __int8 *a2)
{
  __int64 v4; // rdx
  int v5; // eax
  __int64 v6; // rsi
  const char *v7; // rax
  __int64 v8; // r14
  _BYTE *v9; // rax
  __int64 v10; // rax
  int v11; // ecx
  int v12; // eax
  int v13; // ecx
  const char *v14; // rax
  int v15; // ecx
  __int64 v16; // r14
  _BYTE *v17; // rax
  _BYTE *v18; // rsi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  _BYTE *v21; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v22[4]; // [rsp+10h] [rbp-50h] BYREF
  char v23; // [rsp+30h] [rbp-30h]
  char v24; // [rsp+31h] [rbp-2Fh]

  v4 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL);
  if ( v4 == *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL) )
  {
    v5 = *a2 - 42;
    v6 = *((_QWORD *)a2 + 1);
    switch ( v5 )
    {
      case 0:
      case 2:
      case 4:
      case 6:
      case 7:
      case 9:
      case 10:
        v11 = *(unsigned __int8 *)(v6 + 8);
        if ( (unsigned int)(v11 - 17) <= 1 )
          LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
        if ( (_BYTE)v11 != 12 )
        {
          v24 = 1;
          v7 = "Integer arithmetic operators only work with integral types!";
          goto LABEL_4;
        }
        if ( v4 == v6 )
          goto LABEL_15;
        v24 = 1;
        v14 = "Integer arithmetic operators must have same type for operands and result!";
        goto LABEL_31;
      case 1:
      case 3:
      case 5:
      case 8:
      case 11:
        v12 = *(unsigned __int8 *)(v6 + 8);
        if ( (unsigned int)(v12 - 17) <= 1 )
          LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
        if ( (unsigned __int8)v12 > 3u && (_BYTE)v12 != 5 && (v12 & 0xFD) != 4 )
        {
          v21 = a2;
          v24 = 1;
          v22[0] = "Floating-point arithmetic operators only work with floating-point types!";
          v23 = 3;
          sub_BEF5C0(a1, (__int64)v22, &v21);
          return;
        }
        if ( v4 == v6 )
          goto LABEL_15;
        v24 = 1;
        v7 = "Floating-point arithmetic operators must have same type for operands and result!";
        goto LABEL_4;
      case 12:
      case 13:
      case 14:
        v13 = *(unsigned __int8 *)(v6 + 8);
        if ( (unsigned int)(v13 - 17) <= 1 )
          LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
        if ( (_BYTE)v13 != 12 )
        {
          v24 = 1;
          v7 = "Shifts only work with integral types!";
          goto LABEL_4;
        }
        if ( v4 == v6 )
          goto LABEL_15;
        v24 = 1;
        v14 = "Shift return type must be same as operands!";
        goto LABEL_31;
      case 15:
      case 16:
      case 17:
        v15 = *(unsigned __int8 *)(v6 + 8);
        if ( (unsigned int)(v15 - 17) <= 1 )
          LOBYTE(v15) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
        if ( (_BYTE)v15 != 12 )
        {
          v24 = 1;
          v7 = "Logical operators only work with integral types!";
          goto LABEL_4;
        }
        if ( v4 == v6 )
        {
LABEL_15:
          sub_BF6FE0((__int64)a1, (__int64)a2);
          return;
        }
        v24 = 1;
        v14 = "Logical operators must have same type for operands and result!";
LABEL_31:
        v16 = *(_QWORD *)a1;
        v22[0] = v14;
        v23 = 3;
        if ( !v16 )
          goto LABEL_10;
        sub_CA0E80(v22, v16);
        v17 = *(_BYTE **)(v16 + 32);
        if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 24) )
        {
          sub_CB5D20(v16, 10);
        }
        else
        {
          *(_QWORD *)(v16 + 32) = v17 + 1;
          *v17 = 10;
        }
        v18 = *(_BYTE **)a1;
        a1[152] = 1;
        if ( v18 )
        {
          if ( *a2 <= 0x1Cu )
          {
            sub_A5C020(a2, (__int64)v18, 1, (__int64)(a1 + 16));
            v19 = *(_QWORD *)a1;
            v20 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
            if ( (unsigned __int64)v20 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
              goto LABEL_37;
          }
          else
          {
            sub_A693B0((__int64)a2, v18, (__int64)(a1 + 16), 0);
            v19 = *(_QWORD *)a1;
            v20 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
            if ( (unsigned __int64)v20 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            {
LABEL_37:
              *(_QWORD *)(v19 + 32) = v20 + 1;
              *v20 = 10;
              return;
            }
          }
          sub_CB5D20(v19, 10);
        }
        break;
      default:
        BUG();
    }
  }
  else
  {
    v24 = 1;
    v7 = "Both operands to a binary operator are not of the same type!";
LABEL_4:
    v8 = *(_QWORD *)a1;
    v22[0] = v7;
    v23 = 3;
    if ( v8 )
    {
      sub_CA0E80(v22, v8);
      v9 = *(_BYTE **)(v8 + 32);
      if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
      {
        sub_CB5D20(v8, 10);
      }
      else
      {
        *(_QWORD *)(v8 + 32) = v9 + 1;
        *v9 = 10;
      }
      v10 = *(_QWORD *)a1;
      a1[152] = 1;
      if ( v10 )
        sub_BDBD80((__int64)a1, a2);
    }
    else
    {
LABEL_10:
      a1[152] = 1;
    }
  }
}
