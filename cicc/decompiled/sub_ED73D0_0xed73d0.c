// Function: sub_ED73D0
// Address: 0xed73d0
//
__int64 __fastcall sub_ED73D0(_QWORD *a1)
{
  __int64 result; // rax
  _QWORD *v2; // r12
  _QWORD *v3; // r14
  _QWORD *v4; // r13
  _QWORD *v5; // rdi
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // rdi
  __int64 v9; // [rsp+0h] [rbp-40h]
  _QWORD *v10; // [rsp+8h] [rbp-38h]

  *a1 = &unk_49E4D18;
  result = a1[1];
  v9 = result;
  if ( result )
  {
    v2 = *(_QWORD **)(result + 32);
    v10 = *(_QWORD **)(result + 40);
    if ( v10 != v2 )
    {
      do
      {
        v3 = (_QWORD *)v2[6];
        if ( v3 )
        {
          v4 = v3 + 9;
          do
          {
            v5 = (_QWORD *)*(v4 - 3);
            v6 = (_QWORD *)*(v4 - 2);
            v4 -= 3;
            v7 = v5;
            if ( v6 != v5 )
            {
              do
              {
                if ( *v7 )
                  j_j___libc_free_0(*v7, v7[2] - *v7);
                v7 += 3;
              }
              while ( v6 != v7 );
              v5 = (_QWORD *)*v4;
            }
            if ( v5 )
              j_j___libc_free_0(v5, v4[2] - (_QWORD)v5);
          }
          while ( v3 != v4 );
          j_j___libc_free_0(v3, 72);
        }
        v8 = v2[3];
        if ( v8 )
          j_j___libc_free_0(v8, v2[5] - v8);
        if ( *v2 )
          j_j___libc_free_0(*v2, v2[2] - *v2);
        v2 += 10;
      }
      while ( v10 != v2 );
      v2 = *(_QWORD **)(v9 + 32);
    }
    if ( v2 )
      j_j___libc_free_0(v2, *(_QWORD *)(v9 + 48) - (_QWORD)v2);
    return j_j___libc_free_0(v9, 80);
  }
  return result;
}
