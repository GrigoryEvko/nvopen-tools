// Function: sub_72D910
// Address: 0x72d910
//
__int64 __fastcall sub_72D910(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  int v7; // edi
  int v8; // r14d
  _BYTE *v9; // rbx
  __int64 v10; // rax
  __int64 result; // rax

  v6 = *(_QWORD *)(a4 + 32);
  *(_QWORD *)(a3 + 48) = v6;
  v7 = *(_DWORD *)(v6 + 164);
  v8 = dword_4F07270[0];
  if ( dword_4F07270[0] == v7 )
    v8 = 0;
  else
    sub_7296B0(v7);
  v9 = sub_7264B0();
  sub_729730(v8);
  v10 = *(_QWORD *)(a4 + 216);
  v9[16] = a2;
  *((_QWORD *)v9 + 4) = a3;
  *(_QWORD *)v9 = v10;
  result = a1;
  *((_QWORD *)v9 + 1) = a1;
  switch ( a2 )
  {
    case 1:
    case 6:
      v9[24] = 6;
      break;
    case 2:
    case 3:
      v9[24] = 2;
      *(_BYTE *)(a3 + 177) |= 0x10u;
      break;
    case 4:
      v9[24] = 6;
      *(_BYTE *)(a3 + 169) |= 4u;
      break;
    case 5:
      v9[24] = 6;
      *(_BYTE *)(a3 + 169) |= 8u;
      break;
    case 7:
      v9[24] = 8;
      *(_BYTE *)(a3 + 146) |= 2u;
      break;
    case 8:
      v9[24] = 2;
      *(_BYTE *)(a3 + 172) |= 4u;
      break;
    default:
      sub_721090();
  }
  *(_QWORD *)(a4 + 216) = v9;
  return result;
}
