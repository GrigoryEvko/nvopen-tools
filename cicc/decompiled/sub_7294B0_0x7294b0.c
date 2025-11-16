// Function: sub_7294B0
// Address: 0x7294b0
//
void __fastcall sub_7294B0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rax

  if ( a1 )
  {
    v2 = *a2;
    *a2 = (__int64)a1;
    if ( v2 )
    {
      do
      {
        v3 = a1;
        a1 = (_QWORD *)*a1;
      }
      while ( a1 );
      *v3 = v2;
    }
  }
}
