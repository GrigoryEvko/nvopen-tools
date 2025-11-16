// Function: sub_2450190
// Address: 0x2450190
//
unsigned __int64 sub_2450190()
{
  if ( (unsigned int)qword_4FE64A8 > (unsigned int)qword_4FE6588 )
    sub_C64ED0("SampledBurstDuration must be less than or equal to SampledPeriod", 1u);
  if ( !(_DWORD)qword_4FE64A8 || !(_DWORD)qword_4FE6588 )
    sub_C64ED0("SampledPeriod and SampledBurstDuration must be greater than 0", 1u);
  return __PAIR64__(qword_4FE6588, qword_4FE64A8);
}
