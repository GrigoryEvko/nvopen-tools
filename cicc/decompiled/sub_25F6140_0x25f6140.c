// Function: sub_25F6140
// Address: 0x25f6140
//
unsigned __int64 __fastcall sub_25F6140(unsigned __int64 **a1, __int64 a2)
{
  unsigned __int64 *v2; // r14
  __int64 *v3; // rax
  __int64 *v4; // rbx
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // r14

  v2 = *a1;
  v3 = (__int64 *)sub_22077B0(0x18u);
  v4 = v3;
  if ( v3 )
    sub_1049690(v3, a2);
  v5 = *v2;
  *v2 = (unsigned __int64)v4;
  if ( v5 )
  {
    v6 = *(_QWORD *)(v5 + 16);
    if ( v6 )
    {
      sub_FDC110(*(__int64 **)(v5 + 16));
      j_j___libc_free_0(v6);
    }
    j_j___libc_free_0(v5);
  }
  return **a1;
}
