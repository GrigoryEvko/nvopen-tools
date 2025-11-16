// Function: sub_22420F0
// Address: 0x22420f0
//
int sub_22420F0()
{
  int result; // eax

  result = get_nprocs();
  if ( result < 0 )
    return 0;
  return result;
}
