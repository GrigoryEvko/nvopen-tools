// Function: sub_D1B960
// Address: 0xd1b960
//
__int64 __fastcall sub_D1B960(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 result; // rax

  v2 = sub_D1B8E0(a1, a2);
  result = 255;
  if ( v2 )
    return ((*v2 & 3) << 6) | (16 * (*v2 & 3)) | *v2 & 3 | (4 * ((unsigned int)*v2 & 3));
  return result;
}
