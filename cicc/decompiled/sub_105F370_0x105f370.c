// Function: sub_105F370
// Address: 0x105f370
//
void __fastcall sub_105F370(_QWORD *a1)
{
  _QWORD *v1; // r13
  void *v2; // r14
  _QWORD *v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rdi
  _QWORD *v6; // rdi
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  _QWORD *v9; // rbx

  if ( a1 )
  {
    v1 = a1;
    v2 = sub_C33340();
    do
    {
      v3 = v1;
      sub_105F370(v1[3]);
      v4 = v1[22];
      v1 = (_QWORD *)v1[2];
      if ( v4 )
        j_j___libc_free_0_0(v4);
      if ( (void *)v3[18] == v2 )
      {
        v8 = (_QWORD *)v3[19];
        if ( v8 )
        {
          v9 = &v8[3 * *(v8 - 1)];
          if ( v8 != v9 )
          {
            do
            {
              v9 -= 3;
              sub_91D830(v9);
            }
            while ( (_QWORD *)v3[19] != v9 );
          }
          j_j_j___libc_free_0_0(v9 - 1);
        }
      }
      else
      {
        sub_C338F0((__int64)(v3 + 18));
      }
      if ( *((_DWORD *)v3 + 34) > 0x40u )
      {
        v5 = v3[16];
        if ( v5 )
          j_j___libc_free_0_0(v5);
      }
      v6 = (_QWORD *)v3[12];
      if ( v6 != v3 + 14 )
        j_j___libc_free_0(v6, v3[14] + 1LL);
      v7 = (_QWORD *)v3[8];
      if ( v7 != v3 + 10 )
        j_j___libc_free_0(v7, v3[10] + 1LL);
      j_j___libc_free_0(v3, 200);
    }
    while ( v1 );
  }
}
