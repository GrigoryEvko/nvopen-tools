// Function: sub_E05870
// Address: 0xe05870
//
const char *__fastcall sub_E05870(int a1)
{
  const char *result; // rax

  result = "DW_CHILDREN_no";
  if ( a1 )
  {
    result = "DW_CHILDREN_yes";
    if ( a1 != 1 )
      return 0;
  }
  return result;
}
