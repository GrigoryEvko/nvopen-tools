// Function: sub_ED7520
// Address: 0xed7520
//
__int64 __fastcall sub_ED7520(_QWORD *a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12
  _QWORD *v3; // r14
  _QWORD *v4; // r13
  _QWORD *v5; // rdi
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // rdi
  __int64 v11; // [rsp+10h] [rbp-40h]
  _QWORD *v12; // [rsp+18h] [rbp-38h]

  *a1 = &unk_49E4D18;
  v1 = a1[1];
  v11 = v1;
  if ( v1 )
  {
    v2 = *(_QWORD **)(v1 + 32);
    v12 = *(_QWORD **)(v1 + 40);
    if ( v12 != v2 )
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
      while ( v12 != v2 );
      v2 = *(_QWORD **)(v11 + 32);
    }
    if ( v2 )
      j_j___libc_free_0(v2, *(_QWORD *)(v11 + 48) - (_QWORD)v2);
    j_j___libc_free_0(v11, 80);
  }
  return j_j___libc_free_0(a1, 56);
}
