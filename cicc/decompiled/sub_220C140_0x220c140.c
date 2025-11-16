// Function: sub_220C140
// Address: 0x220c140
//
void __fastcall sub_220C140(_QWORD *a1)
{
  _QWORD *v2; // rbp
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  const wchar_t *v5; // r13
  unsigned __int64 v6; // rdi

  v2 = (_QWORD *)a1[2];
  *a1 = off_4A06110;
  if ( v2[3] )
  {
    v3 = v2[2];
    if ( v3 )
    {
      j_j___libc_free_0_0(v3);
      v2 = (_QWORD *)a1[2];
    }
  }
  if ( v2[9] )
  {
    v4 = v2[8];
    if ( v4 )
    {
      j_j___libc_free_0_0(v4);
      v2 = (_QWORD *)a1[2];
    }
  }
  if ( v2[11] )
  {
    v5 = (const wchar_t *)v2[10];
    if ( wcscmp(v5, "(") )
    {
      if ( v5 )
      {
        j_j___libc_free_0_0((unsigned __int64)v5);
        v2 = (_QWORD *)a1[2];
      }
    }
  }
  if ( !v2[7] || (v6 = v2[6]) == 0 || (j_j___libc_free_0_0(v6), (v2 = (_QWORD *)a1[2]) != 0) )
    (*(void (__fastcall **)(_QWORD *))(*v2 + 8LL))(v2);
  nullsub_801();
}
