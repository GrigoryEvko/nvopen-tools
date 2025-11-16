// Function: sub_260C430
// Address: 0x260c430
//
_QWORD *__fastcall sub_260C430(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  _QWORD *v3; // r13
  _QWORD *v4; // r12
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  unsigned __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-58h]
  __int64 v13; // [rsp+10h] [rbp-50h]
  unsigned __int64 v15; // [rsp+20h] [rbp-40h]

  v3 = a3;
  v13 = (__int64)a2 - a1;
  v12 = 0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 3);
  if ( (__int64)a2 - a1 <= 0 )
    return a3;
  v15 = 0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 3);
  v4 = a2;
  do
  {
    v4 -= 3;
    v5 = *(v3 - 3);
    v3 -= 3;
    v6 = v3[1];
    *v3 = *v4;
    v3[1] = v4[1];
    v3[2] = v4[2];
    *v4 = 0;
    v4[1] = 0;
    v4[2] = 0;
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
    --v15;
  }
  while ( v15 );
  v10 = 0x1FFFFFFFFFFFFFFDLL * v12;
  if ( v13 <= 0 )
    v10 = 0x1FFFFFFFFFFFFFFDLL;
  return &a3[v10];
}
