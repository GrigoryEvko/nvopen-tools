// Function: sub_220D180
// Address: 0x220d180
//
void __fastcall sub_220D180(_QWORD *a1)
{
  _QWORD *v2; // rbp
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  _BYTE *v5; // rdi
  unsigned __int64 v6; // rdi
  void (__fastcall *v7)(unsigned __int64); // rax

  v2 = (_QWORD *)a1[2];
  *a1 = off_4A06CD8;
  if ( v2[3] )
  {
    v3 = v2[2];
    if ( v3 )
    {
      j_j___libc_free_0_0(v3);
      v2 = (_QWORD *)a1[2];
    }
  }
  if ( v2[8] )
  {
    v4 = v2[7];
    if ( v4 )
    {
      j_j___libc_free_0_0(v4);
      v2 = (_QWORD *)a1[2];
    }
  }
  if ( v2[10] )
  {
    v5 = (_BYTE *)v2[9];
    if ( *v5 != 40 || v5[1] != 41 || v5[2] )
    {
      j_j___libc_free_0_0((unsigned __int64)v5);
      v2 = (_QWORD *)a1[2];
    }
  }
  if ( !v2[6] || (v6 = v2[5]) == 0 || (j_j___libc_free_0_0(v6), (v2 = (_QWORD *)a1[2]) != 0) )
  {
    v7 = *(void (__fastcall **)(unsigned __int64))(*v2 + 8LL);
    if ( v7 == sub_220C3C0 )
    {
      sub_220C360((__int64)v2);
      j___libc_free_0((unsigned __int64)v2);
    }
    else
    {
      v7((unsigned __int64)v2);
    }
  }
  nullsub_801();
}
