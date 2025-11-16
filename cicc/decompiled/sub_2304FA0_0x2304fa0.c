// Function: sub_2304FA0
// Address: 0x2304fa0
//
void __fastcall sub_2304FA0(_QWORD *a1)
{
  __int64 *v1; // r12

  v1 = (__int64 *)a1[3];
  *a1 = &unk_4A0AED0;
  if ( v1 )
  {
    sub_FDC110(v1);
    j_j___libc_free_0((unsigned __int64)v1);
  }
}
