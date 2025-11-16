// Function: sub_725130
// Address: 0x725130
//
void __fastcall sub_725130(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 *v2; // rax

  v1 = qword_4F07970;
  if ( a1 )
  {
    while ( 1 )
    {
      v2 = (__int64 *)*a1;
      *a1 = v1;
      v1 = (__int64)a1;
      qword_4F07970 = (__int64)a1;
      if ( !v2 )
        break;
      a1 = v2;
    }
  }
}
