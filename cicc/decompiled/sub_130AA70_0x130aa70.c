// Function: sub_130AA70
// Address: 0x130aa70
//
__int64 __fastcall sub_130AA70(int a1, char *a2, size_t a3)
{
  char *v4; // rax

  v4 = strerror_r(a1, a2, a3);
  if ( a2 != v4 )
  {
    strncpy(a2, v4, a3);
    a2[a3 - 1] = 0;
  }
  return 0;
}
