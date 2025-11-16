// Function: sub_130CC80
// Address: 0x130cc80
//
int __fastcall sub_130CC80(void *a1, void *a2)
{
  int result; // eax

  if ( a1 )
    result = mprotect(a1, 0x1000u, 0);
  if ( a2 )
    return mprotect(a2, 0x1000u, 0);
  return result;
}
