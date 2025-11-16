// Function: sub_220EE20
// Address: 0x220ee20
//
void __fastcall sub_220EE20(_QWORD *a1)
{
  _QWORD *v2; // r12
  unsigned __int64 v3; // rdi
  void (__fastcall *v4)(unsigned __int64); // rax

  v2 = (_QWORD *)a1[2];
  *a1 = off_4A07980;
  if ( !v2[3] || (v3 = v2[2]) == 0 || (j_j___libc_free_0_0(v3), (v2 = (_QWORD *)a1[2]) != 0) )
  {
    v4 = *(void (__fastcall **)(unsigned __int64))(*v2 + 8LL);
    if ( v4 == sub_220E740 )
    {
      sub_220E6F0((__int64)v2);
      j___libc_free_0((unsigned __int64)v2);
    }
    else
    {
      v4((unsigned __int64)v2);
    }
  }
  nullsub_801();
}
