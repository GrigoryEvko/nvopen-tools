// Function: sub_1A53FD0
// Address: 0x1a53fd0
//
__int64 __fastcall sub_1A53FD0(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  _QWORD *v10; // rcx
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 *v13; // rax
  unsigned __int64 *v14; // r15
  __int64 v16; // [rsp+8h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v2 = ((((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
      | (a1[3] + 2LL)
      | (((unsigned __int64)a1[3] + 2) >> 1)) >> 4;
  v3 = ((v2
       | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
       | (a1[3] + 2LL)
       | (((unsigned __int64)a1[3] + 2) >> 1)) >> 8)
     | v2
     | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
     | (a1[3] + 2LL)
     | (((unsigned __int64)a1[3] + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v16 = malloc(16 * v4);
  if ( !v16 )
    sub_16BD1C0("Allocation failed", 1u);
  v6 = 16LL * a1[2];
  v7 = *(_QWORD *)a1 + v6;
  if ( *(_QWORD *)a1 != v7 )
  {
    v8 = (_QWORD *)v16;
    v9 = (_QWORD *)(*(_QWORD *)a1 + 8LL);
    v10 = (_QWORD *)(v16 + v6);
    do
    {
      if ( v8 )
      {
        *v8 = *(v9 - 1);
        v8[1] = *v9;
        *v9 = 0;
      }
      v8 += 2;
      v9 += 2;
    }
    while ( v8 != v10 );
    v11 = *(_QWORD *)a1;
    v7 = *(_QWORD *)a1 + 16LL * a1[2];
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v12 = *(_QWORD *)(v7 - 8);
        v7 -= 16LL;
        if ( (v12 & 4) != 0 )
        {
          v13 = (unsigned __int64 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
          v14 = v13;
          if ( v13 )
          {
            if ( (unsigned __int64 *)*v13 != v13 + 2 )
              _libc_free(*v13);
            j_j___libc_free_0(v14, 48);
          }
        }
      }
      while ( v7 != v11 );
      v7 = *(_QWORD *)a1;
    }
  }
  if ( (unsigned int *)v7 != a1 + 4 )
    _libc_free(v7);
  a1[3] = v4;
  *(_QWORD *)a1 = v16;
  return v16;
}
