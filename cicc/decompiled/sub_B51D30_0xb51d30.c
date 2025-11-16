// Function: sub_B51D30
// Address: 0xb51d30
//
__int64 __fastcall sub_B51D30(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int16 a6)
{
  __int64 v9; // rax
  __int64 v10; // r12
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
  __int64 v23; // rax

  switch ( a1 )
  {
    case '&':
      v13 = sub_BD2C40(72, unk_3F10A14);
      v10 = v13;
      if ( v13 )
        sub_B51510(v13, a2, a3, a4, a5, a6);
      break;
    case '\'':
      v14 = sub_BD2C40(72, unk_3F10A14);
      v10 = v14;
      if ( v14 )
        sub_B515B0(v14, a2, a3, a4, a5, a6);
      break;
    case '(':
      v15 = sub_BD2C40(72, unk_3F10A14);
      v10 = v15;
      if ( v15 )
        sub_B51650(v15, a2, a3, a4, a5, a6);
      break;
    case ')':
      v16 = sub_BD2C40(72, unk_3F10A14);
      v10 = v16;
      if ( v16 )
        sub_B51970(v16, a2, a3, a4, a5, a6);
      break;
    case '*':
      v17 = sub_BD2C40(72, unk_3F10A14);
      v10 = v17;
      if ( v17 )
        sub_B51A10(v17, a2, a3, a4, a5, a6);
      break;
    case '+':
      v18 = sub_BD2C40(72, unk_3F10A14);
      v10 = v18;
      if ( v18 )
        sub_B51830(v18, a2, a3, a4, a5, a6);
      break;
    case ',':
      v19 = sub_BD2C40(72, unk_3F10A14);
      v10 = v19;
      if ( v19 )
        sub_B518D0(v19, a2, a3, a4, a5, a6);
      break;
    case '-':
      v20 = sub_BD2C40(72, unk_3F10A14);
      v10 = v20;
      if ( v20 )
        sub_B516F0(v20, a2, a3, a4, a5, a6);
      break;
    case '.':
      v21 = sub_BD2C40(72, unk_3F10A14);
      v10 = v21;
      if ( v21 )
        sub_B51790(v21, a2, a3, a4, a5, a6);
      break;
    case '/':
      v22 = sub_BD2C40(72, unk_3F10A14);
      v10 = v22;
      if ( v22 )
        sub_B51AB0(v22, a2, a3, a4, a5, a6);
      break;
    case '0':
      v23 = sub_BD2C40(72, unk_3F10A14);
      v10 = v23;
      if ( v23 )
        sub_B51B50(v23, a2, a3, a4, a5, a6);
      break;
    case '1':
      v9 = sub_BD2C40(72, unk_3F10A14);
      v10 = v9;
      if ( v9 )
        sub_B51BF0(v9, a2, a3, a4, a5, a6);
      break;
    case '2':
      v12 = sub_BD2C40(72, unk_3F10A14);
      v10 = v12;
      if ( v12 )
        sub_B51C90(v12, a2, a3, a4, a5, a6);
      break;
    default:
      BUG();
  }
  return v10;
}
