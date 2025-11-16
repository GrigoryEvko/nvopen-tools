// Function: sub_C70380
// Address: 0xc70380
//
__int64 __fastcall sub_C70380(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // ebx
  int v7; // eax
  unsigned int v8; // edx
  unsigned __int8 v9; // bl
  bool v10; // cl

  v6 = *(_DWORD *)(a4 + 24);
  if ( v6 <= 0x40 )
  {
    v8 = *(_DWORD *)(a4 + 8);
    v9 = *(_QWORD *)(a4 + 16) != 0;
    if ( v8 > 0x40 )
      goto LABEL_3;
LABEL_6:
    v10 = *(_QWORD *)a4 == 0;
    goto LABEL_4;
  }
  v7 = sub_C444A0(a4 + 16);
  v8 = *(_DWORD *)(a4 + 8);
  v9 = v6 != v7;
  if ( v8 <= 0x40 )
    goto LABEL_6;
LABEL_3:
  v10 = v8 == (unsigned int)sub_C444A0(a4);
LABEL_4:
  sub_C6EF30(a1, a2, a3, !v10, v9);
  return a1;
}
