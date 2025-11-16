// Function: sub_30072B0
// Address: 0x30072b0
//
__int64 __fastcall sub_30072B0(__int64 a1)
{
  unsigned int v1; // ebx
  unsigned __int16 v2; // dx
  _QWORD *v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rdx
  char v6; // al
  unsigned int v7; // eax
  __int16 v9; // ax
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  char v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h]
  __int64 v14; // [rsp+18h] [rbp-18h]

  v2 = *(_WORD *)a1;
  v3 = **(_QWORD ***)(a1 + 8);
  if ( *(_WORD *)a1 )
  {
    if ( v2 == 1 || (unsigned __int16)(v2 - 504) <= 7u )
      BUG();
    v10 = 16LL * (v2 - 1);
    v5 = *(_QWORD *)&byte_444C4A0[v10];
    v6 = byte_444C4A0[v10 + 8];
  }
  else
  {
    v13 = sub_3007260(a1);
    v14 = v4;
    v5 = v13;
    v6 = v14;
  }
  v11 = v5;
  v12 = v6;
  v7 = sub_CA1930(&v11);
  switch ( v7 )
  {
    case 1u:
      v9 = 2;
      break;
    case 2u:
      v9 = 3;
      break;
    case 4u:
      v9 = 4;
      break;
    case 8u:
      v9 = 5;
      break;
    case 0x10u:
      v9 = 6;
      break;
    case 0x20u:
      v9 = 7;
      break;
    case 0x40u:
      v9 = 8;
      break;
    case 0x80u:
      v9 = 9;
      break;
    default:
      return (unsigned int)sub_3007020(v3, v7);
  }
  LOWORD(v1) = v9;
  return v1;
}
