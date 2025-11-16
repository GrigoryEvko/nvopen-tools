// Function: sub_39BC050
// Address: 0x39bc050
//
void __fastcall sub_39BC050(char *src, char *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( a2 - src <= 112 )
  {
    sub_39BAD70(src, a2);
  }
  else
  {
    v2 = (a2 - src) >> 4;
    v3 = &src[8 * v2];
    v4 = (8 * v2) >> 3;
    sub_39BC050(src);
    sub_39BC050(v3);
    sub_39BBF00((__int64)src, (__int64)v3, (__int64)a2, v4, (a2 - v3) >> 3);
  }
}
