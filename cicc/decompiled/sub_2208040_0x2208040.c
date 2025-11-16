// Function: sub_2208040
// Address: 0x2208040
//
__int64 sub_2208040()
{
  __int64 result; // rax
  struct _IO_FILE *v1; // r12
  struct _IO_FILE *v2; // r12
  struct _IO_FILE *v3; // r12
  struct _IO_FILE *v4; // r12
  struct _IO_FILE *v5; // r12
  struct _IO_FILE *v6; // r12

  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd((volatile signed __int32 *)&unk_4FD6918, 1u);
    if ( (_DWORD)result )
      return result;
  }
  else
  {
    result = unk_4FD6918;
    ++unk_4FD6918;
    if ( (_DWORD)result )
      return result;
  }
  qword_4FD45E8 = 0;
  qword_4FD45F0 = 0;
  unk_4CDFAA0 = 1;
  qword_4FD45F8 = 0;
  qword_4FD4600 = 0;
  v1 = stdout;
  qword_4FD4608 = 0;
  qword_4FD4610 = 0;
  qword_4FD45E0 = (__int64)off_4A07480;
  sub_220A990(&unk_4FD4618);
  qword_4FD4620 = (__int64)v1;
  qword_4FD4580 = (__int64)off_4A07480;
  dword_4FD4628 = -1;
  qword_4FD45E0 = (__int64)off_4A06300;
  qword_4FD4588 = 0;
  v2 = stdin;
  qword_4FD4590 = 0;
  qword_4FD4598 = 0;
  qword_4FD45A0 = 0;
  qword_4FD45A8 = 0;
  qword_4FD45B0 = 0;
  sub_220A990(&unk_4FD45B8);
  qword_4FD45C0 = (__int64)v2;
  qword_4FD4520 = (__int64)off_4A07480;
  v3 = stderr;
  qword_4FD4580 = (__int64)off_4A06300;
  dword_4FD45C8 = -1;
  qword_4FD4528 = 0;
  qword_4FD4530 = 0;
  qword_4FD4538 = 0;
  qword_4FD4540 = 0;
  qword_4FD4548 = 0;
  qword_4FD4550 = 0;
  sub_220A990(&unk_4FD4558);
  qword_4FD4520 = (__int64)off_4A06300;
  qword_4FD4560 = (__int64)v3;
  dword_4FD4568 = -1;
  sub_222DF20(&qword_4FD4D00[1]);
  LOWORD(qword_4FD4D00[29]) = 0;
  qword_4FD4D00[28] = 0;
  qword_4FD4D00[0] = &unk_4A06F78;
  qword_4FD4D00[30] = 0;
  qword_4FD4D00[31] = 0;
  qword_4FD4D00[32] = 0;
  qword_4FD4D00[33] = 0;
  qword_4FD4D00[1] = &unk_4A06FA0;
  sub_222DD70(&qword_4FD4D00[1], &qword_4FD45E0);
  sub_222DF20(qword_4FD4E30);
  qword_4FD4E30[27] = 0;
  LOWORD(qword_4FD4E30[28]) = 0;
  qword_4FD4E30[29] = 0;
  qword_4FD4E30[30] = 0;
  qword_4FD4E30[31] = 0;
  qword_4FD4E30[32] = 0;
  unk_4FD4E20 = &unk_4A068A8;
  qword_4FD4E30[0] = &unk_4A068D0;
  unk_4FD4E28 = 0;
  sub_222DD70(qword_4FD4E30, &qword_4FD4580);
  sub_222DF20(&qword_4FD4BE0[1]);
  qword_4FD4BE0[0] = &unk_4A06F78;
  qword_4FD4BE0[28] = 0;
  LOWORD(qword_4FD4BE0[29]) = 0;
  qword_4FD4BE0[30] = 0;
  qword_4FD4BE0[31] = 0;
  qword_4FD4BE0[32] = 0;
  qword_4FD4BE0[33] = 0;
  qword_4FD4BE0[1] = &unk_4A06FA0;
  sub_222DD70(&qword_4FD4BE0[1], &qword_4FD4520);
  sub_222DF20(qword_4FD4AC8);
  qword_4FD4AC8[27] = 0;
  LOWORD(qword_4FD4AC8[28]) = 0;
  unk_4FD4AC0 = &unk_4A06F78;
  qword_4FD4AC8[29] = 0;
  qword_4FD4AC8[30] = 0;
  qword_4FD4AC8[31] = 0;
  qword_4FD4AC8[32] = 0;
  qword_4FD4AC8[0] = &unk_4A06FA0;
  sub_222DD70(qword_4FD4AC8, &qword_4FD4520);
  LODWORD(qword_4FD4BE0[4]) |= 0x2000u;
  qword_4FD4E30[27] = qword_4FD4D00;
  qword_4FD4BE0[28] = qword_4FD4D00;
  v4 = stdout;
  qword_4FD44C8 = 0;
  qword_4FD44D0 = 0;
  qword_4FD44D8 = 0;
  qword_4FD44C0 = (__int64)off_4A07500;
  qword_4FD44E0 = 0;
  qword_4FD44E8 = 0;
  qword_4FD44F0 = 0;
  sub_220A990(&unk_4FD44F8);
  qword_4FD4500 = (__int64)v4;
  qword_4FD4460 = (__int64)off_4A07500;
  dword_4FD4508 = -1;
  qword_4FD44C0 = (__int64)off_4A06380;
  qword_4FD4468 = 0;
  v5 = stdin;
  qword_4FD4470 = 0;
  qword_4FD4478 = 0;
  qword_4FD4480 = 0;
  qword_4FD4488 = 0;
  qword_4FD4490 = 0;
  sub_220A990(&unk_4FD4498);
  qword_4FD44A0 = (__int64)v5;
  qword_4FD4400 = (__int64)off_4A07500;
  v6 = stderr;
  qword_4FD4460 = (__int64)off_4A06380;
  dword_4FD44A8 = -1;
  qword_4FD4408 = 0;
  qword_4FD4410 = 0;
  qword_4FD4418 = 0;
  qword_4FD4420 = 0;
  qword_4FD4428 = 0;
  qword_4FD4430 = 0;
  sub_220A990(&unk_4FD4438);
  qword_4FD4400 = (__int64)off_4A06380;
  qword_4FD4440 = (__int64)v6;
  dword_4FD4448 = -1;
  sub_222DF20(&qword_4FD4880[1]);
  BYTE4(qword_4FD4880[29]) = 0;
  qword_4FD4880[28] = 0;
  LODWORD(qword_4FD4880[29]) = 0;
  qword_4FD4880[0] = &unk_4A06FD8;
  qword_4FD4880[30] = 0;
  qword_4FD4880[31] = 0;
  qword_4FD4880[32] = 0;
  qword_4FD4880[33] = 0;
  qword_4FD4880[1] = &unk_4A07000;
  sub_222DEC0(&qword_4FD4880[1], &qword_4FD44C0);
  sub_222DF20(qword_4FD49B0);
  qword_4FD49B0[27] = 0;
  BYTE4(qword_4FD49B0[28]) = 0;
  LODWORD(qword_4FD49B0[28]) = 0;
  qword_4FD49B0[29] = 0;
  qword_4FD49B0[30] = 0;
  qword_4FD49B0[31] = 0;
  qword_4FD49B0[32] = 0;
  unk_4FD49A0 = &unk_4A06908;
  qword_4FD49B0[0] = &unk_4A06930;
  unk_4FD49A8 = 0;
  sub_222DEC0(qword_4FD49B0, &qword_4FD4460);
  sub_222DF20(qword_4FD4768);
  BYTE4(qword_4FD4768[28]) = 0;
  unk_4FD4760 = &unk_4A06FD8;
  qword_4FD4768[27] = 0;
  LODWORD(qword_4FD4768[28]) = 0;
  qword_4FD4768[29] = 0;
  qword_4FD4768[30] = 0;
  qword_4FD4768[31] = 0;
  qword_4FD4768[32] = 0;
  qword_4FD4768[0] = &unk_4A07000;
  sub_222DEC0(qword_4FD4768, &qword_4FD4400);
  sub_222DF20(qword_4FD4648);
  qword_4FD4648[27] = 0;
  unk_4FD4640 = &unk_4A06FD8;
  LODWORD(qword_4FD4648[28]) = 0;
  BYTE4(qword_4FD4648[28]) = 0;
  qword_4FD4648[29] = 0;
  qword_4FD4648[30] = 0;
  qword_4FD4648[31] = 0;
  qword_4FD4648[32] = 0;
  qword_4FD4648[0] = &unk_4A07000;
  result = sub_222DEC0(qword_4FD4648, &qword_4FD4400);
  LODWORD(qword_4FD4768[3]) |= 0x2000u;
  qword_4FD49B0[27] = qword_4FD4880;
  qword_4FD4768[27] = qword_4FD4880;
  if ( &_pthread_key_create )
    _InterlockedAdd((volatile signed __int32 *)&unk_4FD6918, 1u);
  else
    ++unk_4FD6918;
  return result;
}
