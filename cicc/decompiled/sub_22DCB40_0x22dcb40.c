// Function: sub_22DCB40
// Address: 0x22dcb40
//
void __fastcall sub_22DCB40(__int64 *a1)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rdi

  v2 = (__int64 *)a1[5];
  v3 = (__int64 *)a1[6];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    sub_22DCB40(v4);
  }
  sub_22DCAC0(a1);
}
