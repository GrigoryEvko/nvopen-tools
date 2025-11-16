// Function: sub_8CD160
// Address: 0x8cd160
//
__int64 __fastcall sub_8CD160(__int64 a1, _QWORD *a2, _DWORD *a3)
{
  __int64 v4; // r12
  _QWORD *v5; // rax
  char v6; // al

  v4 = a1;
  sub_8CBB20(6u, a1, a2);
  v5 = *(_QWORD **)(a1 + 32);
  if ( !v5 || a1 == *v5 )
  {
    v4 = (__int64)a2;
    if ( a3 )
      *a3 = 1;
  }
  v6 = *(_BYTE *)(v4 + 140);
  if ( (unsigned __int8)(v6 - 9) <= 2u )
  {
    sub_8CAE10(v4);
  }
  else
  {
    if ( v6 != 2 || (*(_BYTE *)(v4 + 161) & 8) == 0 )
      sub_721090();
    sub_8CA420(v4);
  }
  return sub_8CD5A0(v4);
}
