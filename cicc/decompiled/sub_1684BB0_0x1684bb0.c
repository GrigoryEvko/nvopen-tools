// Function: sub_1684BB0
// Address: 0x1684bb0
//
__int64 __fastcall sub_1684BB0(__int64 a1)
{
  int v2; // eax
  unsigned int v3; // r12d

  sub_1684B50(&qword_4F9F360);
  if ( !dword_4F9F34C++ )
  {
    if ( (unsigned int)dword_4F9F350 < (unsigned __int64)(a1 + 4096) || (v2 = dword_4F9F348, --dword_4F9F348, !v2) )
    {
      sub_1688C60(qword_4F9F358, 1);
      qword_4F9F358 = sub_1688B40(a1 + 4096, 1);
      if ( !qword_4F9F358 )
      {
        --dword_4F9F34C;
        v3 = 0;
        dword_4F9F350 = 0;
        goto LABEL_5;
      }
      dword_4F9F348 = 100;
      dword_4F9F350 = a1 + 4096;
    }
  }
  v3 = 1;
LABEL_5:
  j__pthread_mutex_unlock(qword_4F9F360);
  return v3;
}
