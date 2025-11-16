// Function: sub_34EAEA0
// Address: 0x34eaea0
//
void __fastcall sub_34EAEA0(unsigned __int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_34E7A40(a1, a2);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = (__int64 *)&a1[v4];
    v6 = (8 * v4) >> 3;
    sub_34EAEA0(a1, v5);
    sub_34EAEA0(v5, a2);
    sub_34EACC0((__int64 *)a1, v5, (__int64)a2, v6, ((char *)a2 - (char *)v5) >> 3, a3);
  }
}
