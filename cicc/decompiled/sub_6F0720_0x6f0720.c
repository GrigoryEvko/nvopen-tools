// Function: sub_6F0720
// Address: 0x6f0720
//
__int64 sub_6F0720()
{
  __int64 v0; // rax
  _QWORD *v1; // rbx
  _QWORD *v2; // rax
  _QWORD *v3; // rcx
  _QWORD *v4; // rdx
  _QWORD *v5; // rbx
  _QWORD *v6; // rax
  _QWORD *v7; // rcx
  _QWORD *v8; // rdx
  _QWORD *v9; // rbx
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  _QWORD *v12; // rdx
  int v13; // edx
  int v14; // edx
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx

  qword_4D03A98 = 0;
  qword_4D03A88 = 0;
  qword_4D03A80 = 0;
  qword_4D03A78 = 0;
  v0 = sub_823970(16);
  qword_4D03A50 = v0;
  if ( v0 )
  {
    v1 = (_QWORD *)v0;
    v2 = (_QWORD *)sub_823970(0x4000);
    v3 = v2;
    v4 = v2 + 2048;
    do
    {
      if ( v2 )
        *v2 = 0;
      v2 += 2;
    }
    while ( v2 != v4 );
    *v1 = v3;
    v1[1] = 1023;
  }
  qword_4D03A70 = sub_823970(16);
  v5 = (_QWORD *)qword_4D03A70;
  if ( qword_4D03A70 )
  {
    v6 = (_QWORD *)sub_823970(0x4000);
    v7 = v6;
    v8 = v6 + 2048;
    do
    {
      if ( v6 )
        *v6 = 0;
      v6 += 2;
    }
    while ( v6 != v8 );
    *v5 = v7;
    v5[1] = 1023;
  }
  qword_4D03A48 = sub_823970(16);
  v9 = (_QWORD *)qword_4D03A48;
  if ( qword_4D03A48 )
  {
    v10 = (_QWORD *)sub_823970(49152);
    v11 = v10;
    v12 = v10 + 6144;
    do
    {
      if ( v10 )
      {
        *v10 = 0;
        v10[1] = 0;
        v10[2] = 0;
        v10[3] = 0;
      }
      v10 += 6;
    }
    while ( v12 != v10 );
    *v9 = v11;
    v9[1] = 1023;
  }
  v13 = unk_4F06948;
  *(_QWORD *)dword_4D04120 = 0xB0000000BLL;
  *(_QWORD *)&dword_4D04120[9] = 0xB00000008LL;
  dword_4D04120[2] = v13;
  *(_QWORD *)&dword_4D04120[11] = 0x3500000018LL;
  v14 = unk_4F0693C;
  *(_QWORD *)&dword_4D04120[13] = 113;
  dword_4D04120[3] = v14;
  dword_4D04120[4] = v14;
  v15 = unk_4F06930;
  dword_4D04120[5] = unk_4F06930;
  dword_4D04120[6] = v15;
  dword_4D04120[7] = unk_4F06924;
  dword_4D04120[8] = unk_4F06918;
  qword_4D040A0[0] = 2;
  qword_4D040A0[2] = unk_4F06A48;
  qword_4D040A0[1] = 2;
  v16 = unk_4F06A38;
  qword_4D040A0[3] = unk_4F06A38;
  qword_4D040A0[4] = v16;
  qword_4D040A0[9] = 2;
  v17 = unk_4F06A28;
  qword_4D040A0[10] = 2;
  qword_4D040A0[11] = 4;
  qword_4D040A0[5] = v17;
  qword_4D040A0[6] = v17;
  qword_4D040A0[12] = 8;
  v18 = unk_4F06A18;
  qword_4D040A0[13] = 16;
  qword_4D040A0[14] = 0;
  qword_4D040A0[7] = v18;
  qword_4D040A0[8] = unk_4F06A08;
  unk_4D04060 = 0xFFFFFFF3FFFFFFF3LL;
  unk_4D04068 = unk_4F06944;
  unk_4D0406C = unk_4F06938;
  unk_4D04070 = unk_4F06938;
  unk_4D04074 = unk_4F0692C;
  unk_4D04078 = unk_4F0692C;
  unk_4D0407C = unk_4F06920;
  unk_4D04084 = 0xFFFFFFF3FFFFFF83LL;
  unk_4D0408C = 0xFFFFFC03FFFFFF83LL;
  unk_4D04094 = 4294950915LL;
  unk_4D04080 = unk_4F06914;
  *(_QWORD *)dword_4D04020 = 0x1000000010LL;
  dword_4D04020[2] = unk_4F06940;
  *(_QWORD *)&dword_4D04020[9] = 0x1000000080LL;
  LODWORD(v18) = unk_4F06934;
  *(_QWORD *)&dword_4D04020[11] = 0x40000000080LL;
  *(_QWORD *)&dword_4D04020[13] = 0x4000;
  dword_4D04020[3] = v18;
  dword_4D04020[4] = v18;
  LODWORD(v18) = unk_4F06928;
  dword_4D04020[5] = unk_4F06928;
  dword_4D04020[6] = v18;
  dword_4D04020[7] = unk_4F0691C;
  dword_4D04020[8] = unk_4F06910;
  return sub_84DA50();
}
