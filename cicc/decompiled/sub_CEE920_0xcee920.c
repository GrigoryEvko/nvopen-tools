// Function: sub_CEE920
// Address: 0xcee920
//
int sub_CEE920()
{
  char v0; // r12
  _BYTE *v1; // rax
  char v2; // r12
  _BYTE *v3; // rax
  char v4; // r12
  _BYTE *v5; // rax
  char v6; // r12
  _BYTE *v7; // rax
  char v8; // r12
  _BYTE *v9; // rax
  char v10; // r12
  _BYTE *v11; // rax
  int v12; // r12d
  _DWORD *v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  int v16; // r12d
  _DWORD *v17; // rax
  int v18; // r12d
  _DWORD *v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  char v22; // al
  char v23; // al
  int v24; // r12d
  _DWORD *v25; // rax
  int result; // eax

  v0 = qword_4F86470[2];
  v1 = (_BYTE *)sub_CEECD0(1, 1);
  *v1 = v0;
  sub_C94E10((__int64)qword_4F86470, v1);
  BYTE1(qword_4F86470[2]) = 0;
  v2 = qword_4F86450[2];
  v3 = (_BYTE *)sub_CEECD0(1, 1);
  *v3 = v2;
  sub_C94E10((__int64)qword_4F86450, v3);
  BYTE1(qword_4F86450[2]) = 0;
  v4 = qword_4F86430[2];
  v5 = (_BYTE *)sub_CEECD0(1, 1);
  *v5 = v4;
  sub_C94E10((__int64)qword_4F86430, v5);
  BYTE1(qword_4F86430[2]) = 0;
  v6 = qword_4F863F0[2];
  v7 = (_BYTE *)sub_CEECD0(1, 1);
  *v7 = v6;
  sub_C94E10((__int64)qword_4F863F0, v7);
  BYTE1(qword_4F863F0[2]) = 0;
  v8 = qword_4F863D0[2];
  v9 = (_BYTE *)sub_CEECD0(1, 1);
  *v9 = v8;
  sub_C94E10((__int64)qword_4F863D0, v9);
  BYTE1(qword_4F863D0[2]) = 0;
  v10 = qword_4F863B0[2];
  v11 = (_BYTE *)sub_CEECD0(1, 1);
  *v11 = v10;
  sub_C94E10((__int64)qword_4F863B0, v11);
  BYTE1(qword_4F863B0[2]) = 0;
  v12 = qword_4F86390[2];
  v13 = (_DWORD *)sub_CEECD0(4, 4);
  *v13 = v12;
  sub_C94E10((__int64)qword_4F86390, v13);
  BYTE4(qword_4F86390[2]) = 0;
  LOBYTE(v12) = qword_4F86330[2];
  v14 = (_BYTE *)sub_CEECD0(1, 1);
  *v14 = v12;
  sub_C94E10((__int64)qword_4F86330, v14);
  BYTE1(qword_4F86330[2]) = 0;
  LOBYTE(v12) = qword_4F86350[2];
  v15 = (_BYTE *)sub_CEECD0(1, 1);
  *v15 = v12;
  sub_C94E10((__int64)qword_4F86350, v15);
  BYTE1(qword_4F86350[2]) = 0;
  v16 = qword_4F862F0[2];
  v17 = (_DWORD *)sub_CEECD0(4, 4);
  *v17 = v16;
  sub_C94E10((__int64)qword_4F862F0, v17);
  BYTE4(qword_4F862F0[2]) = 0;
  v18 = qword_4F862D0[2];
  v19 = (_DWORD *)sub_CEECD0(4, 4);
  *v19 = v18;
  sub_C94E10((__int64)qword_4F862D0, v19);
  BYTE4(qword_4F862D0[2]) = 0;
  LOBYTE(v18) = qword_4F862B0[2];
  v20 = (_BYTE *)sub_CEECD0(1, 1);
  *v20 = v18;
  sub_C94E10((__int64)qword_4F862B0, v20);
  BYTE1(qword_4F862B0[2]) = 0;
  LOBYTE(v18) = qword_4F86410[2];
  v21 = (_BYTE *)sub_CEECD0(1, 1);
  *v21 = v18;
  sub_C94E10((__int64)qword_4F86410, v21);
  BYTE1(qword_4F86410[2]) = 0;
  sub_C5DA10((__int64)&qword_4F85600);
  v22 = BYTE1(qword_4F85698);
  if ( BYTE1(qword_4F85698) )
    v22 = qword_4F85698;
  LOBYTE(qword_4F85688) = v22;
  sub_C5DA10((__int64)&qword_4F85520);
  v23 = BYTE1(qword_4F855B8);
  if ( BYTE1(qword_4F855B8) )
    v23 = qword_4F855B8;
  v24 = qword_4F86250[2];
  LOBYTE(qword_4F855A8) = v23;
  v25 = (_DWORD *)sub_CEECD0(4, 4);
  *v25 = v24;
  result = sub_C94E10((__int64)qword_4F86250, v25);
  BYTE4(qword_4F86250[2]) = 0;
  return result;
}
