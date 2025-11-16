// Function: sub_C73140
// Address: 0xc73140
//
__int64 __fastcall sub_C73140(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rcx
  int v7; // eax
  __int64 v8; // rdx
  unsigned int v9; // r14d
  int v10; // eax
  unsigned int v11; // edx
  unsigned __int8 v12; // r14
  bool v13; // cl

  v6 = *(_QWORD *)(a3 + 16);
  v7 = *(_DWORD *)(a3 + 8);
  v8 = *(_QWORD *)a3;
  *(_QWORD *)a3 = v6;
  LODWORD(v6) = *(_DWORD *)(a3 + 24);
  *(_QWORD *)(a3 + 16) = v8;
  *(_DWORD *)(a3 + 8) = v6;
  *(_DWORD *)(a3 + 24) = v7;
  v9 = *(_DWORD *)(a4 + 8);
  if ( v9 <= 0x40 )
  {
    v11 = *(_DWORD *)(a4 + 24);
    v12 = *(_QWORD *)a4 != 0;
    if ( v11 > 0x40 )
      goto LABEL_3;
LABEL_6:
    v13 = *(_QWORD *)(a4 + 16) == 0;
    goto LABEL_4;
  }
  v10 = sub_C444A0(a4);
  v11 = *(_DWORD *)(a4 + 24);
  v12 = v9 != v10;
  if ( v11 <= 0x40 )
    goto LABEL_6;
LABEL_3:
  v13 = v11 == (unsigned int)sub_C444A0(a4 + 16);
LABEL_4:
  sub_C6EF30(a1, a2, a3, !v13, v12);
  return a1;
}
