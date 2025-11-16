// Function: sub_37230A0
// Address: 0x37230a0
//
void __fastcall sub_37230A0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_3722890(a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)a1) >> 4;
    v3 = (char *)&a1[v2];
    v4 = (8 * v2) >> 3;
    sub_37230A0(a1, v3);
    sub_37230A0(v3, a2);
    sub_3722F20((char *)a1, v3, (__int64)a2, v4, ((char *)a2 - v3) >> 3);
  }
}
