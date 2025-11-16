// Function: ctor_003
// Address: 0x439500
//
int ctor_003()
{
  __int64 *v0; // rbx
  __int64 v1; // rax

  qword_4F5FC60 = 0;
  qword_4F5FC70 = 0;
  qword_4F5FC78 = 0;
  qword_4F5FC80 = 0;
  qword_4F5FC88 = 0;
  qword_4F5FC90 = 0;
  qword_4F5FC98 = 0;
  qword_4F5FCA0 = 0;
  qword_4F5FCA8 = 0;
  qword_4F5FC68 = 8;
  qword_4F5FC60 = sub_22077B0(64);
  v0 = (__int64 *)(qword_4F5FC60 + ((4 * qword_4F5FC68 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v1 = sub_22077B0(512);
  qword_4F5FC88 = v0;
  *v0 = v1;
  qword_4F5FC80 = v1 + 512;
  qword_4F5FCA0 = v1 + 512;
  qword_4F5FCA8 = (__int64)v0;
  qword_4F5FC78 = v1;
  qword_4F5FC98 = v1;
  qword_4F5FC70 = v1;
  qword_4F5FC90 = v1;
  __cxa_atexit(sub_8567A0, &qword_4F5FC60, &qword_4A427C0);
  qword_4F5FC28 = 1;
  qword_4F5FC20 = (__int64)&qword_4F5FC50;
  qword_4F5FC30 = 0;
  qword_4F5FC38 = 0;
  dword_4F5FC40 = 1065353216;
  qword_4F5FC48 = 0;
  qword_4F5FC50 = 0;
  return __cxa_atexit(sub_8565C0, &qword_4F5FC50 - 6, &qword_4A427C0);
}
