// Function: sub_130B160
// Address: 0x130b160
//
__int64 __fastcall sub_130B160(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // [rsp+8h] [rbp-28h] BYREF
  struct timespec tp; // [rsp+10h] [rbp-20h] BYREF

  sub_130B140(&v2, a1);
  clock_gettime(1, &tp);
  sub_130B0D0(a1, tp.tv_sec, tp.tv_nsec);
  result = sub_130B150(&v2, a1);
  if ( (int)result > 0 )
    return sub_130B140(a1, &v2);
  return result;
}
