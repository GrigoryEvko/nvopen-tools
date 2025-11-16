// Function: sub_39BBE80
// Address: 0x39bbe80
//
void __fastcall sub_39BBE80(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_39BB3B0(a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)a1) >> 4;
    v3 = (char *)&a1[v2];
    v4 = (8 * v2) >> 3;
    sub_39BBE80(a1, v3);
    sub_39BBE80(v3, a2);
    sub_39BBD00((char *)a1, v3, (__int64)a2, v4, ((char *)a2 - v3) >> 3);
  }
}
