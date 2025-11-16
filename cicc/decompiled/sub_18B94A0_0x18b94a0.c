// Function: sub_18B94A0
// Address: 0x18b94a0
//
unsigned __int64 __fastcall sub_18B94A0(__int64 a1, unsigned __int64 a2)
{
  char *v2; // r15
  unsigned __int64 result; // rax
  char *v5; // r12
  _QWORD *v6; // r13
  _QWORD *v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  signed __int64 v15; // [rsp+8h] [rbp-38h]

  if ( a2 > 0x124924924924924LL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *(char **)a1;
  result = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)a1) >> 4);
  if ( a2 > result )
  {
    v5 = *(char **)(a1 + 8);
    v6 = 0;
    result = v5 - v2;
    v15 = v5 - v2;
    if ( a2 )
    {
      result = sub_22077B0(112 * a2);
      v5 = *(char **)(a1 + 8);
      v2 = *(char **)a1;
      v6 = (_QWORD *)result;
    }
    if ( v5 != v2 )
    {
      v7 = v6;
      do
      {
        if ( v7 )
        {
          *v7 = *(_QWORD *)v2;
          v7[1] = *((_QWORD *)v2 + 1);
          v7[2] = *((_QWORD *)v2 + 2);
          v7[3] = *((_QWORD *)v2 + 3);
          v7[4] = *((_QWORD *)v2 + 4);
          v8 = *((_QWORD *)v2 + 5);
          *((_QWORD *)v2 + 4) = 0;
          *((_QWORD *)v2 + 3) = 0;
          *((_QWORD *)v2 + 2) = 0;
          v7[5] = v8;
          v7[6] = *((_QWORD *)v2 + 6);
          v7[7] = *((_QWORD *)v2 + 7);
          v9 = *((_QWORD *)v2 + 8);
          *((_QWORD *)v2 + 7) = 0;
          *((_QWORD *)v2 + 6) = 0;
          *((_QWORD *)v2 + 5) = 0;
          v7[8] = v9;
          v7[9] = *((_QWORD *)v2 + 9);
          v7[10] = *((_QWORD *)v2 + 10);
          v10 = *((_QWORD *)v2 + 11);
          *((_QWORD *)v2 + 10) = 0;
          *((_QWORD *)v2 + 9) = 0;
          *((_QWORD *)v2 + 8) = 0;
          v7[11] = v10;
          v7[12] = *((_QWORD *)v2 + 12);
          v7[13] = *((_QWORD *)v2 + 13);
          *((_QWORD *)v2 + 13) = 0;
          *((_QWORD *)v2 + 11) = 0;
        }
        else
        {
          v14 = *((_QWORD *)v2 + 11);
          if ( v14 )
            result = j_j___libc_free_0(v14, *((_QWORD *)v2 + 13) - v14);
        }
        v11 = *((_QWORD *)v2 + 8);
        if ( v11 )
          result = j_j___libc_free_0(v11, *((_QWORD *)v2 + 10) - v11);
        v12 = *((_QWORD *)v2 + 5);
        if ( v12 )
          result = j_j___libc_free_0(v12, *((_QWORD *)v2 + 7) - v12);
        v13 = *((_QWORD *)v2 + 2);
        if ( v13 )
          result = j_j___libc_free_0(v13, *((_QWORD *)v2 + 4) - v13);
        v2 += 112;
        v7 += 14;
      }
      while ( v2 != v5 );
      v2 = *(char **)a1;
    }
    if ( v2 )
      result = j_j___libc_free_0(v2, *(_QWORD *)(a1 + 16) - (_QWORD)v2);
    *(_QWORD *)a1 = v6;
    *(_QWORD *)(a1 + 8) = (char *)v6 + v15;
    *(_QWORD *)(a1 + 16) = &v6[14 * a2];
  }
  return result;
}
