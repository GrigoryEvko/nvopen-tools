// Function: sub_B8FDC0
// Address: 0xb8fdc0
//
void __fastcall sub_B8FDC0(char *a1, unsigned int *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - a1 <= 224 )
  {
    sub_B8F7D0((__int64)a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - a1) >> 5;
    v3 = &a1[16 * v2];
    v4 = (16 * v2) >> 4;
    sub_B8FDC0(a1, v3);
    sub_B8FDC0(v3, a2);
    sub_B8FC60(a1, v3, (__int64)a2, v4, ((char *)a2 - v3) >> 4);
  }
}
