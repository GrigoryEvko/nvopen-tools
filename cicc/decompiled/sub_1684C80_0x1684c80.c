// Function: sub_1684C80
// Address: 0x1684c80
//
__int64 __fastcall sub_1684C80(unsigned __int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3; // rax

  if ( !a1 )
    return 0;
  if ( !unk_4CD28E0
    || !*(_QWORD *)(sub_1689050() + 96)
    || (v3 = sub_1689050(), (result = sub_16870F0(*(_QWORD *)(v3 + 96), a1 >> 3)) == 0) )
  {
    if ( qword_4F9F338 )
    {
      sub_1684B50(&qword_4F9F330);
      v1 = sub_16870F0(qword_4F9F338, a1 >> 3);
      j__pthread_mutex_unlock(qword_4F9F330);
      return v1;
    }
    return 0;
  }
  return result;
}
