// Function: sub_323CF60
// Address: 0x323cf60
//
void __fastcall sub_323CF60(__int64 a1)
{
  unsigned __int16 v2; // ax
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax

  v2 = sub_3220AA0(a1);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 <= 4u )
  {
    v5 = sub_31DA6B0(v3);
    sub_323CCF0(a1, *(_QWORD *)(v5 + 144));
  }
  else
  {
    v4 = sub_31DA6B0(v3);
    sub_323CCF0(a1, *(_QWORD *)(v4 + 328));
  }
}
