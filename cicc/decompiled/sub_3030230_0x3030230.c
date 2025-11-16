// Function: sub_3030230
// Address: 0x3030230
//
__int64 __fastcall sub_3030230(__int128 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int128 v7; // rax
  int v8; // r8d
  int v9; // r9d
  int v10; // r12d
  int v11; // r13d
  __int128 v12; // rax
  int v13; // r9d

  *(_QWORD *)&v7 = sub_30301B0(a2);
  if ( !(_QWORD)v7 )
    return 0;
  v10 = v9;
  v11 = v8;
  *(_QWORD *)&v12 = sub_3406EB0(*(_QWORD *)(a6 + 16), 58, v9, a3, v8, v9, a1, v7);
  return sub_3406EB0(*(_QWORD *)(a6 + 16), 56, v10, a3, v11, v13, v12, a1);
}
