// Function: sub_220B560
// Address: 0x220b560
//
void __fastcall sub_220B560(_QWORD *a1)
{
  unsigned __int64 *v2; // rdi
  _BYTE *v3; // r8

  *a1 = off_4A05728;
  v2 = (unsigned __int64 *)a1[2];
  if ( v2[3] && v2[2] )
  {
    j_j___libc_free_0_0(v2[2]);
    v2 = (unsigned __int64 *)a1[2];
  }
  if ( v2[8] && v2[7] )
  {
    j_j___libc_free_0_0(v2[7]);
    v2 = (unsigned __int64 *)a1[2];
  }
  if ( v2[10] )
  {
    v3 = (_BYTE *)v2[9];
    if ( *v3 != 40 || v3[1] != 41 || v3[2] )
    {
      j_j___libc_free_0_0(v2[9]);
      v2 = (unsigned __int64 *)a1[2];
    }
  }
  if ( !v2[6] || !v2[5] || (j_j___libc_free_0_0(v2[5]), (v2 = (unsigned __int64 *)a1[2]) != 0) )
    (*(void (__fastcall **)(unsigned __int64 *))(*v2 + 8))(v2);
  nullsub_801();
}
