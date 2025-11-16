// Function: sub_D0EF00
// Address: 0xd0ef00
//
void __fastcall sub_D0EF00(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r15
  _QWORD *v3; // r14
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rax
  _QWORD *v7; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_D0EF00(v1[3]);
      v3 = (_QWORD *)v1[5];
      v1 = (_QWORD *)v1[2];
      if ( v3 )
      {
        v4 = v3[3];
        v5 = v3[2];
        if ( v4 != v5 )
        {
          do
          {
            while ( 1 )
            {
              if ( *(_BYTE *)(v5 + 24) )
              {
                v6 = *(_QWORD *)(v5 + 16);
                *(_BYTE *)(v5 + 24) = 0;
                if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
                  break;
              }
              v5 += 40;
              if ( v4 == v5 )
                goto LABEL_11;
            }
            v7 = (_QWORD *)v5;
            v5 += 40;
            sub_BD60C0(v7);
          }
          while ( v4 != v5 );
LABEL_11:
          v5 = v3[2];
        }
        if ( v5 )
          j_j___libc_free_0(v5, v3[4] - v5);
        j_j___libc_free_0(v3, 48);
      }
      j_j___libc_free_0(v2, 48);
    }
    while ( v1 );
  }
}
