// Function: sub_335D700
// Address: 0x335d700
//
void __fastcall sub_335D700(char *a1, unsigned int *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - a1 <= 224 )
  {
    sub_335C220((__int64)a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - a1) >> 5;
    v3 = &a1[16 * v2];
    v4 = (16 * v2) >> 4;
    sub_335D700(a1, v3);
    sub_335D700(v3, a2);
    sub_335D5A0(a1, v3, (__int64)a2, v4, ((char *)a2 - v3) >> 4);
  }
}
