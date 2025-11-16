// Function: sub_386FCA0
// Address: 0x386fca0
//
void __fastcall sub_386FCA0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 224 )
  {
    sub_386FA70(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 5;
    v5 = &a1[2 * v4];
    v6 = (16 * v4) >> 4;
    sub_386FCA0(a1, v5);
    sub_386FCA0(v5, a2);
    sub_386F620(a1, v5, (__int64)a2, v6, ((char *)a2 - (char *)v5) >> 4, a3);
  }
}
