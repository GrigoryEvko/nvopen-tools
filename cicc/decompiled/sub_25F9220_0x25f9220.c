// Function: sub_25F9220
// Address: 0x25f9220
//
void __fastcall sub_25F9220(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_25F79D0(a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)a1) >> 4;
    v3 = &a1[v2];
    v4 = (8 * v2) >> 3;
    sub_25F9220(a1, v3);
    sub_25F9220(v3, a2);
    sub_25F9040((char *)a1, v3, (__int64)a2, v4, a2 - v3);
  }
}
