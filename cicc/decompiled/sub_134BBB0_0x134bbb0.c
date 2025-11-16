// Function: sub_134BBB0
// Address: 0x134bbb0
//
__int64 __fastcall sub_134BBB0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  __int64 result; // rax

  a1[384] += a2[384];
  v2 = a1 + 384;
  a1[385] += a2[385];
  a1[386] += a2[386];
  a1[387] += a2[387];
  a1[388] += a2[388];
  a1[389] += a2[389];
  a1[390] += a2[390];
  a1[391] += a2[391];
  a1[392] += a2[392];
  a1[393] += a2[393];
  a1[394] += a2[394];
  a1[395] += a2[395];
  do
  {
    *a1 += *a2;
    a2 += 6;
    a1[1] += *(a2 - 5);
    a1[2] += *(a2 - 4);
    a1[3] += *(a2 - 3);
    a1[4] += *(a2 - 2);
    result = *(a2 - 1);
    a1[5] += result;
    a1 += 6;
  }
  while ( a1 != v2 );
  return result;
}
