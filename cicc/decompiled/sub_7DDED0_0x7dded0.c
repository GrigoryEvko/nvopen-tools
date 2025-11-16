// Function: sub_7DDED0
// Address: 0x7dded0
//
void __fastcall sub_7DDED0(__int64 a1, _QWORD *a2)
{
  __int64 **v2; // rbx

  dword_4F189C0 = 1;
  sub_7DD9B0(unk_4F07288, a2);
  v2 = (__int64 **)qword_4F072C0;
  if ( qword_4F072C0 )
  {
    do
    {
      sub_7DD8B0((__m128i *)v2[3], a2);
      v2 = (__int64 **)*v2;
    }
    while ( v2 );
  }
  sub_7DD8B0((__m128i *)qword_4F07300, a2);
  dword_4F189C0 = 0;
}
