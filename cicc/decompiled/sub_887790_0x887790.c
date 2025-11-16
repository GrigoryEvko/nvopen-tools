// Function: sub_887790
// Address: 0x887790
//
__int64 sub_887790()
{
  int v0; // edx
  __int64 v1; // rdx

  v0 = dword_4F077C4;
  *(_QWORD *)dword_4F04BA0 = 0x300000004LL;
  *(_QWORD *)&dword_4F04BA0[2] = 0x200000002LL;
  if ( v0 == 2 )
  {
    *(_QWORD *)&dword_4F04BA0[4] = 0x200000002LL;
    *(_QWORD *)&dword_4F04BA0[6] = 0x200000002LL;
  }
  else
  {
    v0 = 6;
    *(_QWORD *)&dword_4F04BA0[4] = 0x100000001LL;
    *(_QWORD *)&dword_4F04BA0[6] = 0x200000001LL;
  }
  dword_4F04BA0[8] = v0;
  dword_4F04BA0[9] = v0;
  dword_4F04BA0[10] = v0;
  *(_QWORD *)&dword_4F04BA0[17] = 0x200000002LL;
  *(_QWORD *)&dword_4F04BA0[19] = 0x200000002LL;
  *(_QWORD *)&dword_4F04BA0[21] = 0x200000002LL;
  *(_QWORD *)&dword_4F04BA0[23] = 0x200000002LL;
  *(_QWORD *)&dword_4F04BA0[13] = 0x500000002LL;
  *(_QWORD *)&dword_4F04BA0[11] = 2;
  v1 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)&dword_4F04BA0[15] = 0x200000005LL;
  dword_4F04BA0[25] = 2;
  qword_4F60028 = 0;
  xmmword_4F06660[0].m128i_i64[0] = 0;
  xmmword_4F06660[1].m128i_i32[0] &= 0xFF800000;
  xmmword_4F06660[0].m128i_i64[1] = v1;
  xmmword_4F06660[1].m128i_i64[1] = 0;
  xmmword_4F06660[2].m128i_i64[0] = 0;
  xmmword_4F06660[2].m128i_i64[1] = 0;
  xmmword_4F06660[3].m128i_i64[0] = 0;
  xmmword_4F06660[3].m128i_i64[1] = 0;
  qword_4F60020 = 0;
  dword_4D049E0 = 0;
  xmmword_4F60040 = 0u;
  xmmword_4F60050 = 0u;
  *(_QWORD *)&xmmword_4F60060 = 0;
  *((_QWORD *)&xmmword_4F60060 + 1) = 0xFFFFFFFFLL;
  *(_DWORD *)((char *)&xmmword_4F60090 + 1) &= 0x10000000u;
  qword_4F066A0 = 0;
  *(_QWORD *)&xmmword_4F60070 = v1;
  DWORD2(xmmword_4F60070) = 0;
  xmmword_4F60080 = 0u;
  qword_4F5FED0 = 0;
  qword_4F5FFC0 = 0;
  if ( unk_4D04508 )
    sub_8539C0((__int64)&off_4B7D660);
  sub_8D0840(&qword_4F60030, 8, 0);
  sub_8D0840(&qword_4D049B8, 8, 0);
  sub_8D0840(&dword_4F5FFBC, 4, 0);
  sub_8D0840(&qword_4D049B0, 8, 0);
  sub_8D0840(&dword_4F5FFB0, 4, 0);
  sub_8D0840(&unk_4D049D8, 8, 0);
  sub_8D0840(&unk_4D049D0, 8, 0);
  sub_8D0840(&unk_4D049C8, 8, 0);
  sub_8D0840(&unk_4D049C0, 8, 0);
  sub_8D0840(&dword_4F600B0, 4, 0);
  sub_8D0840(&qword_4D04998, 8, 0);
  sub_8D0840(&dword_4D04988, 8, 0);
  sub_8D0840(&qword_4D04980, 8, 0);
  sub_8D0840(&unk_4D04978, 8, 0);
  sub_8D0840(&qword_4D04970, 8, 0);
  sub_8D0840(&qword_4F600B8, 8, 0);
  sub_8D0840(&unk_4D04968, 4, 0);
  sub_8D0840(&unk_4D04A48, 8, 0);
  sub_8D0840(&dword_4F066AC, 4, 0);
  sub_8D0840(&qword_4F5FFC8, 8, 0);
  sub_8D0840(&unk_4F066A8, 4, 0);
  sub_8D0840(&qword_4D04A00, 64, 0);
  return sub_8D0840(&qword_4F5FED8, 8, 0);
}
