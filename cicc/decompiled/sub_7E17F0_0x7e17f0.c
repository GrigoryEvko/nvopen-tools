// Function: sub_7E17F0
// Address: 0x7e17f0
//
void __fastcall sub_7E17F0(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 *v2; // rax

  v1 = qword_4F18A20;
  if ( a1 )
  {
    while ( 1 )
    {
      v2 = (__int64 *)*a1;
      *a1 = v1;
      v1 = (__int64)a1;
      qword_4F18A20 = (__int64)a1;
      if ( !v2 )
        break;
      a1 = v2;
    }
  }
}
