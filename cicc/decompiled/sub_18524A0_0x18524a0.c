// Function: sub_18524A0
// Address: 0x18524a0
//
__int64 __fastcall sub_18524A0(_QWORD *a1)
{
  _QWORD *v1; // r15
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rdi
  __int64 v7; // rdi
  _QWORD *v8; // rdi
  __int64 result; // rax
  _QWORD *i; // [rsp+8h] [rbp-38h]

  for ( i = a1; i; result = j_j___libc_free_0(v1, 152) )
  {
    v1 = i;
    sub_18524A0(i[3]);
    v2 = i[15];
    i = (_QWORD *)i[2];
    while ( v2 )
    {
      v3 = v2;
      sub_18523F0(*(_QWORD **)(v2 + 24));
      v4 = *(_QWORD *)(v2 + 96);
      v2 = *(_QWORD *)(v2 + 16);
      while ( v4 )
      {
        v5 = v4;
        sub_1852120(*(_QWORD **)(v4 + 24));
        v6 = *(_QWORD *)(v4 + 32);
        v4 = *(_QWORD *)(v4 + 16);
        if ( v6 )
          j_j___libc_free_0(v6, *(_QWORD *)(v5 + 48) - v6);
        j_j___libc_free_0(v5, 80);
      }
      v7 = *(_QWORD *)(v3 + 48);
      if ( v7 != v3 + 64 )
        j_j___libc_free_0(v7, *(_QWORD *)(v3 + 64) + 1LL);
      j_j___libc_free_0(v3, 128);
    }
    v8 = (_QWORD *)v1[4];
    if ( v8 != v1 + 6 )
      j_j___libc_free_0(v8, v1[6] + 1LL);
  }
  return result;
}
