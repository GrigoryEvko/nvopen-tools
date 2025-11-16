// Function: sub_603B70
// Address: 0x603b70
//
_DWORD *sub_603B70()
{
  _QWORD *v0; // rbx
  _DWORD *v1; // rax
  _DWORD *v2; // rcx
  _DWORD *v3; // rdx
  _DWORD *result; // rax
  _QWORD *v5; // rbx
  _DWORD *v6; // rcx
  _DWORD *v7; // rdx

  qword_4CF8000 = 0;
  qword_4CF7FF8 = 0;
  qword_4CF7FD8 = 0;
  qword_4CF7FD0 = 0;
  qword_4CF7FE0 = 0;
  qword_4CF7FC0 = 0;
  v0 = (_QWORD *)sub_823970(16);
  unk_4CF7FF0 = v0;
  if ( v0 )
  {
    v1 = (_DWORD *)sub_823970(0x4000);
    v2 = v1;
    v3 = v1 + 4096;
    do
    {
      if ( v1 )
        *v1 = 0;
      v1 += 4;
    }
    while ( v1 != v3 );
    *v0 = v2;
    v0[1] = 1023;
  }
  result = (_DWORD *)sub_823970(16);
  qword_4CF7FE8 = (__int64)result;
  v5 = result;
  if ( result )
  {
    result = (_DWORD *)sub_823970(0x4000);
    v6 = result;
    v7 = result + 4096;
    do
    {
      if ( result )
        *result = 0;
      result += 4;
    }
    while ( v7 != result );
    *v5 = v6;
    v5[1] = 1023;
  }
  return result;
}
