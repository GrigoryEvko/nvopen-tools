// Function: sub_2D93020
// Address: 0x2d93020
//
_QWORD **__fastcall sub_2D93020(_QWORD **a1, const void *a2, size_t a3, char a4)
{
  const char *v5; // r12
  size_t v6; // rax

  v5 = "true";
  if ( !a4 )
    v5 = "false";
  v6 = strlen(v5);
  return sub_A78980(a1, a2, a3, v5, v6);
}
