// Function: sub_8192C0
// Address: 0x8192c0
//
__int64 __fastcall sub_8192C0(_DWORD *a1, unsigned int *a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  _BYTE *v6; // rdi
  __int64 v7; // rdx
  __int64 *v8; // rax

  while ( 1 )
  {
    unk_4F063E4 = 0;
    dword_4F063E0 = 0;
    unk_4F063DC = 0;
    sub_7BC390();
    *a1 = 0;
    if ( dword_4F063EC )
    {
      if ( dword_4F063EC != 1 || (v3 = dword_4D04954) == 0 )
        *a1 = 1;
    }
    v6 = qword_4F06460;
    v7 = dword_4F063E0;
    qword_4F194D0 = (__int64)qword_4F06460;
    if ( !dword_4F063E0
      || (unsigned __int64)qword_4F06460 >= qword_4F06498 && (unsigned __int64)qword_4F06460 < qword_4F06490 )
    {
      break;
    }
    a2 = 0;
    v8 = sub_7AEFF0((unsigned __int64)qword_4F06460);
    v7 = qword_4F194C8;
    if ( v8[10] <= (unsigned __int64)qword_4F194C8 )
    {
      if ( dword_4F194C0 )
        break;
    }
    ++qword_4F06460;
  }
  return sub_7B8B50((unsigned __int64)v6, a2, v7, v3, v4, v5);
}
