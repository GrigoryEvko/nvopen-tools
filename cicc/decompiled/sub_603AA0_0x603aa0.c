// Function: sub_603AA0
// Address: 0x603aa0
//
__int64 sub_603AA0()
{
  __int64 v0; // rbx
  __int64 v1; // rdi
  __int64 result; // rax

  v0 = qword_4CF7F90;
  while ( v0 )
  {
    sub_5E9730(*(_QWORD *)(v0 + 24));
    v1 = v0;
    v0 = *(_QWORD *)(v0 + 16);
    j_j___libc_free_0(v1, 40);
  }
  qword_4CF7F90 = 0;
  qword_4CF7F98 = (__int64)&dword_4CF7F88;
  qword_4CF7FA0 = (__int64)&dword_4CF7F88;
  qword_4CF7FA8 = 0;
  qword_4CF8008 = 0;
  result = dword_4F077BC;
  qword_4CF7FB0 = 0;
  if ( dword_4F077BC )
    result = qword_4F077A8 <= 0x76BFu;
  dword_4CF8020 = result;
  qword_4CF8018 = 0;
  qword_4CF8010 = 0;
  qword_4CF7FC8 = 0;
  return result;
}
