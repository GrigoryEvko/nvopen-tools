// Function: sub_15FDBD0
// Address: 0x15fdbd0
//
__int64 __fastcall sub_15FDBD0(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax

  switch ( a1 )
  {
    case '$':
      v12 = sub_1648A60(56, 1);
      v9 = v12;
      if ( v12 )
        sub_15FC510(v12, a2, a3, a4, a5);
      break;
    case '%':
      v13 = sub_1648A60(56, 1);
      v9 = v13;
      if ( v13 )
        sub_15FC690(v13, a2, a3, a4, a5);
      break;
    case '&':
      v14 = sub_1648A60(56, 1);
      v9 = v14;
      if ( v14 )
        sub_15FC810(v14, a2, a3, a4, a5);
      break;
    case '\'':
      v15 = sub_1648A60(56, 1);
      v9 = v15;
      if ( v15 )
        sub_15FCF90(v15, a2, a3, a4, a5);
      break;
    case '(':
      v16 = sub_1648A60(56, 1);
      v9 = v16;
      if ( v16 )
        sub_15FD110(v16, a2, a3, a4, a5);
      break;
    case ')':
      v17 = sub_1648A60(56, 1);
      v9 = v17;
      if ( v17 )
        sub_15FCC90(v17, a2, a3, a4, a5);
      break;
    case '*':
      v18 = sub_1648A60(56, 1);
      v9 = v18;
      if ( v18 )
        sub_15FCE10(v18, a2, a3, a4, a5);
      break;
    case '+':
      v19 = sub_1648A60(56, 1);
      v9 = v19;
      if ( v19 )
        sub_15FC990(v19, a2, a3, a4, a5);
      break;
    case ',':
      v20 = sub_1648A60(56, 1);
      v9 = v20;
      if ( v20 )
        sub_15FCB10(v20, a2, a3, a4, a5);
      break;
    case '-':
      v21 = sub_1648A60(56, 1);
      v9 = v21;
      if ( v21 )
        sub_15FD290(v21, a2, a3, a4, a5);
      break;
    case '.':
      v22 = sub_1648A60(56, 1);
      v9 = v22;
      if ( v22 )
        sub_15FD410(v22, a2, a3, a4, a5);
      break;
    case '/':
      v8 = sub_1648A60(56, 1);
      v9 = v8;
      if ( v8 )
        sub_15FD590(v8, a2, a3, a4, a5);
      break;
    case '0':
      v11 = sub_1648A60(56, 1);
      v9 = v11;
      if ( v11 )
        sub_15FDB10(v11, a2, a3, a4, a5);
      break;
  }
  return v9;
}
