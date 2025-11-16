// Function: sub_18321F0
// Address: 0x18321f0
//
__int64 sub_18321F0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 v2; // rax

  v0 = sub_22077B0(280);
  v1 = v0;
  if ( v0 )
  {
    sub_186A7A0(v0, &unk_4FAA49C);
    *(_QWORD *)(v1 + 260) = 0;
    *(_WORD *)(v1 + 256) = 256;
    *(_QWORD *)(v1 + 268) = 0;
    *(_BYTE *)(v1 + 153) = 1;
    *(_QWORD *)v1 = off_49F0B90;
    v2 = sub_163A1D0();
    sub_18320F0(v2);
  }
  return v1;
}
