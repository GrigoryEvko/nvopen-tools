// Function: sub_105F4D0
// Address: 0x105f4d0
//
void __fastcall sub_105F4D0(_QWORD *a1)
{
  _QWORD *v1; // r13
  void *v2; // r14
  _QWORD *v3; // r12
  _QWORD *v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  _QWORD *v7; // rdi
  _QWORD *v8; // rdi
  _QWORD *v9; // rax
  _QWORD *v10; // rbx

  if ( a1 )
  {
    v1 = a1;
    v2 = sub_C33340();
    do
    {
      v3 = v1;
      sub_105F4D0(v1[3]);
      v4 = (_QWORD *)v1[26];
      v1 = (_QWORD *)v1[2];
      sub_105F370(v4);
      v5 = v3[22];
      if ( v5 )
        j_j___libc_free_0_0(v5);
      if ( (void *)v3[18] == v2 )
      {
        v9 = (_QWORD *)v3[19];
        if ( v9 )
        {
          v10 = &v9[3 * *(v9 - 1)];
          if ( v9 != v10 )
          {
            do
            {
              v10 -= 3;
              sub_91D830(v10);
            }
            while ( (_QWORD *)v3[19] != v10 );
          }
          j_j_j___libc_free_0_0(v10 - 1);
        }
      }
      else
      {
        sub_C338F0((__int64)(v3 + 18));
      }
      if ( *((_DWORD *)v3 + 34) > 0x40u )
      {
        v6 = v3[16];
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      v7 = (_QWORD *)v3[12];
      if ( v7 != v3 + 14 )
        j_j___libc_free_0(v7, v3[14] + 1LL);
      v8 = (_QWORD *)v3[8];
      if ( v8 != v3 + 10 )
        j_j___libc_free_0(v8, v3[10] + 1LL);
      j_j___libc_free_0(v3, 240);
    }
    while ( v1 );
  }
}
