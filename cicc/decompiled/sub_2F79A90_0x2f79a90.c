// Function: sub_2F79A90
// Address: 0x2f79a90
//
void __fastcall sub_2F79A90(_QWORD *a1)
{
  unsigned __int64 v2; // r14
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi

  v2 = a1[22];
  *a1 = &unk_4A2B760;
  if ( v2 )
  {
    v3 = *(unsigned int *)(v2 + 24);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v2 + 8);
      v5 = &v4[4 * v3];
      do
      {
        if ( *v4 != -8192 && *v4 != -4096 )
        {
          v6 = v4[1];
          if ( v6 )
            j_j___libc_free_0(v6);
        }
        v4 += 4;
      }
      while ( v5 != v4 );
      v3 = *(unsigned int *)(v2 + 24);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 8), 32 * v3, 8);
    j_j___libc_free_0(v2);
  }
  sub_BB9280((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
