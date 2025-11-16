// Function: sub_323E1C0
// Address: 0x323e1c0
//
void __fastcall sub_323E1C0(__int64 a1)
{
  __int64 v1; // r13
  unsigned __int16 v3; // ax
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax

  v1 = a1 + 3776;
  if ( !*(_BYTE *)(a1 + 3769) )
    v1 = a1 + 3080;
  v3 = sub_3220AA0(a1);
  v4 = *(_QWORD *)(a1 + 8);
  if ( v3 <= 4u )
  {
    v6 = sub_31DA6B0(v4);
    sub_323DFB0(a1, v1, *(_QWORD *)(v6 + 160));
  }
  else
  {
    v5 = sub_31DA6B0(v4);
    sub_323DFB0(a1, v1, *(_QWORD *)(v5 + 320));
  }
}
