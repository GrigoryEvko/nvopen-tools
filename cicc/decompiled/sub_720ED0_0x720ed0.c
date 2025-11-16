// Function: sub_720ED0
// Address: 0x720ed0
//
__int64 __fastcall sub_720ED0(FILE *stream, int *a2)
{
  unsigned int v2; // r13d

  v2 = 0;
  *a2 = 0;
  if ( stream )
  {
    v2 = fflush(stream);
    if ( v2 )
    {
      v2 = 1;
      *a2 = *__errno_location();
    }
    if ( ferror(stream) )
    {
      v2 = 1;
      *a2 = *__errno_location();
    }
    if ( stdout != stream && fclose(stream) && !v2 )
    {
      v2 = 1;
      *a2 = *__errno_location();
    }
  }
  return v2;
}
