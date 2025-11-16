// Function: sub_5D4320
// Address: 0x5d4320
//
__int64 __fastcall sub_5D4320(__int64 a1, __int64 a2)
{
  int v4; // edi
  char *v5; // rbx

  sub_74A390(a1, a2, 0, 0, 0, &qword_4CF7CE0);
  if ( (_DWORD)a2 )
  {
    v4 = 32;
    v5 = "*";
    do
    {
      ++v5;
      putc(v4, stream);
      v4 = *(v5 - 1);
    }
    while ( *(v5 - 1) );
    dword_4CF7F40 += 2;
  }
  return sub_74D110(a1, (unsigned int)a2, 0, &qword_4CF7CE0);
}
