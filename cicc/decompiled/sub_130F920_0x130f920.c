// Function: sub_130F920
// Address: 0x130f920
//
__int64 sub_130F920()
{
  unsigned __int64 v0; // rsi
  unsigned __int64 v1; // rax

  v0 = qword_4C6F130[0];
  if ( qword_4C6F130[0] < 0LL )
  {
    v1 = 0;
    v0 = 0;
  }
  else
  {
    if ( qword_4C6F130[0] <= 0LL )
      v0 = 1;
    v1 = v0 >> 6;
    if ( !(v0 >> 6) )
      v1 = 1;
    if ( v1 > (unsigned __int64)&dword_400000 )
      v1 = (unsigned __int64)&dword_400000;
  }
  qword_4F969F0 = v1;
  qword_4F96A00 = 0;
  qword_4F96A08 = v0;
  return 0;
}
