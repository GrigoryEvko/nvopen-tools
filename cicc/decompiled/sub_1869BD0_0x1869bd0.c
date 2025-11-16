// Function: sub_1869BD0
// Address: 0x1869bd0
//
__int64 sub_1869BD0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 v2; // rax

  v0 = sub_22077B0(352);
  v1 = v0;
  if ( v0 )
  {
    sub_186A7A0(v0, &unk_4FAB3FC);
    *(_QWORD *)(v1 + 260) = 0;
    *(_WORD *)(v1 + 256) = 0;
    *(_QWORD *)v1 = off_49F1698;
    *(_QWORD *)(v1 + 268) = 0;
    sub_3851190(v1 + 276);
    v2 = sub_163A1D0();
    sub_1869A50(v2);
  }
  return v1;
}
