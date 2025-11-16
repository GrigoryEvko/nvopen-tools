// Function: sub_68ACF0
// Address: 0x68acf0
//
__int64 __fastcall sub_68ACF0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rdi

  v2 = (__int64 *)sub_73E830();
  v3 = *v2;
  *(__int64 *)((char *)v2 + 28) = *(_QWORD *)dword_4F07508;
  if ( (unsigned int)sub_8D32E0(v3) )
  {
    v2 = (__int64 *)sub_73DDB0(v2);
    *(__int64 *)((char *)v2 + 28) = *(_QWORD *)dword_4F07508;
  }
  return sub_6E7170(v2, a2);
}
