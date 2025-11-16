// Function: sub_ED1160
// Address: 0xed1160
//
__int64 __fastcall sub_ED1160(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  _QWORD *v5; // rcx
  _QWORD *v6; // r14
  _QWORD *v7; // rdi
  _QWORD *v8; // r13
  _QWORD *v9; // rbx
  _QWORD *v10; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 48);
  if ( !v2 )
  {
    v2 = sub_22077B0(72);
    if ( v2 )
    {
      *(_QWORD *)(v2 + 64) = 0;
      *(_OWORD *)v2 = 0;
      *(_OWORD *)(v2 + 16) = 0;
      *(_OWORD *)(v2 + 32) = 0;
      *(_OWORD *)(v2 + 48) = 0;
    }
    v5 = *(_QWORD **)(a1 + 48);
    *(_QWORD *)(a1 + 48) = v2;
    v10 = v5;
    v6 = v5 + 9;
    if ( v5 )
    {
      do
      {
        v7 = (_QWORD *)*(v6 - 3);
        v8 = (_QWORD *)*(v6 - 2);
        v6 -= 3;
        v9 = v7;
        if ( v8 != v7 )
        {
          do
          {
            if ( *v9 )
              j_j___libc_free_0(*v9, v9[2] - *v9);
            v9 += 3;
          }
          while ( v8 != v9 );
          v7 = (_QWORD *)*v6;
        }
        if ( v7 )
          j_j___libc_free_0(v7, v6[2] - (_QWORD)v7);
      }
      while ( v10 != v6 );
      j_j___libc_free_0(v10, 72);
      v2 = *(_QWORD *)(a1 + 48);
    }
  }
  return v2 + 24LL * a2;
}
