// Function: sub_29B95F0
// Address: 0x29b95f0
//
void __fastcall sub_29B95F0(__int64 a1, unsigned __int64 a2)
{
  char *v2; // r15
  char *v4; // r12
  _QWORD *v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // r14
  __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  signed __int64 v11; // [rsp+8h] [rbp-38h]

  if ( a2 > 0x124924924924924LL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *(char **)a1;
  if ( a2 > 0x6DB6DB6DB6DB6DB7LL * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)a1) >> 4) )
  {
    v4 = *(char **)(a1 + 8);
    v5 = 0;
    v11 = v4 - v2;
    if ( a2 )
    {
      v6 = sub_22077B0(112 * a2);
      v4 = *(char **)(a1 + 8);
      v2 = *(char **)a1;
      v5 = (_QWORD *)v6;
    }
    if ( v4 != v2 )
    {
      v7 = v5;
      do
      {
        if ( v7 )
        {
          *v7 = *(_QWORD *)v2;
          v7[1] = *((_QWORD *)v2 + 1);
          v7[2] = *((_QWORD *)v2 + 2);
          v7[3] = *((_QWORD *)v2 + 3);
          v7[4] = *((_QWORD *)v2 + 4);
          v7[5] = *((_QWORD *)v2 + 5);
          v7[6] = *((_QWORD *)v2 + 6);
          v7[7] = *((_QWORD *)v2 + 7);
          v7[8] = *((_QWORD *)v2 + 8);
          v7[9] = *((_QWORD *)v2 + 9);
          v7[10] = *((_QWORD *)v2 + 10);
          v8 = *((_QWORD *)v2 + 11);
          *((_QWORD *)v2 + 10) = 0;
          *((_QWORD *)v2 + 9) = 0;
          *((_QWORD *)v2 + 8) = 0;
          v7[11] = v8;
          v7[12] = *((_QWORD *)v2 + 12);
          v7[13] = *((_QWORD *)v2 + 13);
          *((_QWORD *)v2 + 13) = 0;
          *((_QWORD *)v2 + 11) = 0;
        }
        else
        {
          v10 = *((_QWORD *)v2 + 11);
          if ( v10 )
            j_j___libc_free_0(v10);
        }
        v9 = *((_QWORD *)v2 + 8);
        if ( v9 )
          j_j___libc_free_0(v9);
        v2 += 112;
        v7 += 14;
      }
      while ( v2 != v4 );
      v2 = *(char **)a1;
    }
    if ( v2 )
      j_j___libc_free_0((unsigned __int64)v2);
    *(_QWORD *)a1 = v5;
    *(_QWORD *)(a1 + 8) = (char *)v5 + v11;
    *(_QWORD *)(a1 + 16) = &v5[14 * a2];
  }
}
