// Function: sub_385C520
// Address: 0x385c520
//
void __fastcall sub_385C520(unsigned int *src, char *a2, _QWORD *a3)
{
  __int64 v4; // rcx
  unsigned int *v5; // r15
  __int64 v6; // rbx

  if ( a2 - (char *)src <= 56 )
  {
    sub_385B9D0(src, a2, a3);
  }
  else
  {
    v4 = (a2 - (char *)src) >> 3;
    v5 = &src[v4];
    v6 = (4 * v4) >> 2;
    sub_385C520(src);
    sub_385C520(v5);
    sub_385C3A0(src, v5, (__int64)a2, v6, (a2 - (char *)v5) >> 2, a3);
  }
}
