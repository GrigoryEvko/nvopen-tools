// Function: sub_1832270
// Address: 0x1832270
//
__int64 __fastcall sub_1832270(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = sub_22077B0(280);
  v2 = v1;
  if ( v1 )
  {
    sub_186A7A0(v1, &unk_4FAA49C);
    *(_QWORD *)(v2 + 260) = 0;
    *(_WORD *)(v2 + 256) = 256;
    *(_QWORD *)(v2 + 268) = 0;
    *(_BYTE *)(v2 + 153) = a1;
    *(_QWORD *)v2 = off_49F0B90;
    v3 = sub_163A1D0();
    sub_18320F0(v3);
  }
  return v2;
}
