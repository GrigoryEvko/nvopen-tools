// Function: sub_16946D0
// Address: 0x16946d0
//
__int64 __fastcall sub_16946D0(__int64 a1, int a2)
{
  __int64 result; // rax
  _QWORD *v4; // rax
  __int64 v5; // r14
  _QWORD **v6; // r12
  _QWORD *v7; // r15
  _QWORD *v8; // rdi
  _QWORD **v9; // r12
  _QWORD *v10; // r15
  _QWORD *v11; // rdi
  _QWORD **v12; // [rsp+8h] [rbp-38h]
  _QWORD **v13; // [rsp+8h] [rbp-38h]

  if ( !*(_QWORD *)(a1 + 24) )
  {
    v4 = (_QWORD *)sub_22077B0(48);
    if ( v4 )
    {
      *v4 = 0;
      v4[1] = 0;
      v4[2] = 0;
      v4[3] = 0;
      v4[4] = 0;
      v4[5] = 0;
    }
    v5 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 24) = v4;
    if ( v5 )
    {
      v6 = *(_QWORD ***)(v5 + 24);
      v12 = *(_QWORD ***)(v5 + 32);
      if ( v12 != v6 )
      {
        do
        {
          v7 = *v6;
          while ( v7 != v6 )
          {
            v8 = v7;
            v7 = (_QWORD *)*v7;
            j_j___libc_free_0(v8, 32);
          }
          v6 += 3;
        }
        while ( v12 != v6 );
        v6 = *(_QWORD ***)(v5 + 24);
      }
      if ( v6 )
        j_j___libc_free_0(v6, *(_QWORD *)(v5 + 40) - (_QWORD)v6);
      v9 = *(_QWORD ***)v5;
      v13 = *(_QWORD ***)(v5 + 8);
      if ( v13 != *(_QWORD ***)v5 )
      {
        do
        {
          v10 = *v9;
          while ( v10 != v9 )
          {
            v11 = v10;
            v10 = (_QWORD *)*v10;
            j_j___libc_free_0(v11, 32);
          }
          v9 += 3;
        }
        while ( v13 != v9 );
        v9 = *(_QWORD ***)v5;
      }
      if ( v9 )
        j_j___libc_free_0(v9, *(_QWORD *)(v5 + 16) - (_QWORD)v9);
      j_j___libc_free_0(v5, 48);
    }
  }
  result = *(_QWORD *)(a1 + 24);
  if ( a2 )
    result += 24;
  return result;
}
