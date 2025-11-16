// Function: sub_DF5FE0
// Address: 0xdf5fe0
//
const char *__fastcall sub_DF5FE0(__int64 a1, int a2)
{
  const char *result; // rax

  result = "Generic::ScalarRC";
  if ( a2 )
  {
    result = "Generic::Unknown Register Class";
    if ( a2 == 1 )
      return "Generic::VectorRC";
  }
  return result;
}
