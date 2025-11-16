// Function: sub_14CF630
// Address: 0x14cf630
//
__int64 __fastcall sub_14CF630(char a1, int a2, __int64 **a3, __int64 a4, _DWORD *a5)
{
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdi

  switch ( a2 )
  {
    case 0:
      v10 = **a3;
      if ( *((_BYTE *)*a3 + 8) == 16 )
      {
        v11 = (*a3)[4];
        v12 = sub_1643320(v10);
        v13 = sub_16463B0(v12, (unsigned int)v11);
      }
      else
      {
        v13 = sub_1643320(v10);
      }
      result = sub_15A0680(v13, 0, 0);
      break;
    case 1:
      *a5 = a1 == 0 ? 34 : 38;
      result = 0;
      break;
    case 2:
      *a5 = 32;
      result = 0;
      break;
    case 3:
      *a5 = a1 == 0 ? 35 : 39;
      result = 0;
      break;
    case 4:
      *a5 = a1 == 0 ? 36 : 40;
      result = 0;
      break;
    case 5:
      *a5 = 33;
      result = 0;
      break;
    case 6:
      *a5 = a1 == 0 ? 37 : 41;
      result = 0;
      break;
    case 7:
      v6 = **a3;
      if ( *((_BYTE *)*a3 + 8) == 16 )
      {
        v7 = (*a3)[4];
        v8 = sub_1643320(v6);
        v9 = sub_16463B0(v8, (unsigned int)v7);
      }
      else
      {
        v9 = sub_1643320(v6);
      }
      result = sub_15A0680(v9, 1, 0);
      break;
  }
  return result;
}
