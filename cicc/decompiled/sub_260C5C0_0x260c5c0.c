// Function: sub_260C5C0
// Address: 0x260c5c0
//
unsigned __int64 *__fastcall sub_260C5C0(unsigned __int64 *a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned __int64 *v3; // r12
  unsigned __int64 *v4; // r13
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v12; // [rsp+0h] [rbp-50h]
  unsigned __int64 v14; // [rsp+10h] [rbp-40h]

  v12 = a2 - (_QWORD)a1;
  v14 = 0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3);
  if ( a2 - (__int64)a1 <= 0 )
    return a3;
  v3 = a1;
  v4 = a3;
  do
  {
    v5 = *v4;
    v6 = v4[1];
    *v4 = *v3;
    v4[1] = v3[1];
    v4[2] = v3[2];
    *v3 = 0;
    v3[1] = 0;
    v3[2] = 0;
    if ( v5 != v6 )
    {
      v7 = v5;
      do
      {
        v8 = *(unsigned int *)(v7 + 144);
        v9 = *(_QWORD *)(v7 + 128);
        v7 += 152LL;
        sub_C7D6A0(v9, 8 * v8, 4);
        sub_C7D6A0(*(_QWORD *)(v7 - 56), 8LL * *(unsigned int *)(v7 - 40), 4);
        sub_C7D6A0(*(_QWORD *)(v7 - 88), 16LL * *(unsigned int *)(v7 - 72), 8);
        sub_C7D6A0(*(_QWORD *)(v7 - 120), 16LL * *(unsigned int *)(v7 - 104), 8);
      }
      while ( v6 != v7 );
    }
    if ( v5 )
      j_j___libc_free_0(v5);
    v3 += 3;
    v4 += 3;
    --v14;
  }
  while ( v14 );
  v10 = 24;
  if ( v12 > 0 )
    v10 = v12;
  return (unsigned __int64 *)((char *)a3 + v10);
}
