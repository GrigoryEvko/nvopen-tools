// Function: sub_1BBD300
// Address: 0x1bbd300
//
__int64 __fastcall sub_1BBD300(__int64 *a1)
{
  if ( (unsigned int)dword_4FB9000 > (unsigned __int64)(0x2E8BA2E8BA2E8BA3LL * ((a1[1] - *a1) >> 4)) )
    return (unsigned int)sub_1BBD240(a1) ^ 1;
  else
    return 0;
}
