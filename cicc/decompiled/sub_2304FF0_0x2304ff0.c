// Function: sub_2304FF0
// Address: 0x2304ff0
//
void __fastcall sub_2304FF0(_QWORD *a1)
{
  __int64 *v1; // r13

  v1 = (__int64 *)a1[3];
  *a1 = &unk_4A0AED0;
  if ( v1 )
  {
    sub_FDC110(v1);
    j_j___libc_free_0((unsigned __int64)v1);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
