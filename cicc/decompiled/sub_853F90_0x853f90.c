// Function: sub_853F90
// Address: 0x853f90
//
_QWORD *__fastcall sub_853F90(__m128i *a1)
{
  __int64 v2; // rdi

  v2 = a1[4].m128i_i64[0];
  if ( v2 && !*(_BYTE *)(v2 + 16) )
  {
    sub_869FD0(v2, (unsigned int)dword_4F04C64);
    a1[4].m128i_i64[0] = 0;
    if ( (a1[4].m128i_i8[8] & 1) == 0 )
      goto LABEL_4;
LABEL_6:
    sub_7AEA70(a1 + 1);
    goto LABEL_4;
  }
  if ( (a1[4].m128i_i8[8] & 1) != 0 )
    goto LABEL_6;
LABEL_4:
  a1->m128i_i64[0] = qword_4D03D28[0];
  qword_4D03D28[0] = a1;
  return qword_4D03D28;
}
