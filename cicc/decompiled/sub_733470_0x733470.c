// Function: sub_733470
// Address: 0x733470
//
_BYTE *__fastcall sub_733470(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  _BYTE *v6; // r12
  _BYTE *v7; // rax
  _BYTE *v8; // r8
  _QWORD *v9; // rax
  _QWORD *v10; // rdx

  v6 = sub_732EF0(qword_4F04C68[0] + 776LL * dword_4F04C58);
  v7 = sub_725B40();
  *((_QWORD *)v7 + 1) = a1;
  v8 = v7;
  *((_QWORD *)v7 + 2) = a2;
  v7[32] = a3;
  *(_QWORD *)(v7 + 36) = *a4;
  *(_BYTE *)(a1 + 169) |= 0x10u;
  v9 = (_QWORD *)*((_QWORD *)v6 + 26);
  if ( v9 )
  {
    do
    {
      v10 = v9;
      v9 = (_QWORD *)*v9;
    }
    while ( v9 );
    *v10 = v8;
  }
  else
  {
    *((_QWORD *)v6 + 26) = v8;
  }
  return v8;
}
