// Function: sub_2149CD0
// Address: 0x2149cd0
//
_QWORD *sub_2149CD0()
{
  _QWORD *v0; // r8

  v0 = (_QWORD *)sub_22077B0(224);
  if ( v0 )
    memset(v0, 0, 0xE0u);
  v0[1] = 101;
  *((_DWORD *)v0 + 4) = 0;
  *v0 = &unk_4311BC0;
  *((_DWORD *)v0 + 24) = 1;
  v0[3] = &off_4985A40;
  v0[14] = 0;
  v0[6] = &unk_4312768;
  v0[15] = 0;
  v0[7] = &unk_4312760;
  v0[16] = 0;
  v0[8] = "ENVREG10";
  v0[17] = 0;
  v0[9] = "Int1Regs";
  v0[18] = 0;
  v0[5] = &unk_4311A20;
  v0[4] = 0x640000000ELL;
  v0[19] = 0;
  v0[10] = &unk_4312758;
  v0[11] = &unk_4312754;
  v0[13] = &unk_43116E0;
  return v0;
}
