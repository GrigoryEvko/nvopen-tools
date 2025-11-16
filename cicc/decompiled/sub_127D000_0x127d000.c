// Function: sub_127D000
// Address: 0x127d000
//
__int64 __fastcall sub_127D000(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // r12
  __int64 v7; // r15
  __int64 v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  _QWORD *v11; // r13
  __int64 v12; // rdx

  v2 = a1;
  if ( !sub_8D3410(a2) )
    return v2;
  v4 = *(_QWORD *)(a2 + 176);
  v5 = sub_8D4050(a2);
  v6 = (_QWORD *)sub_127D000(a1, v5);
  v7 = sub_1645D80(*v6, v4);
  if ( v4 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v8 = v4;
  if ( v4 )
  {
    v9 = (_QWORD *)sub_22077B0(8 * v4);
    v10 = &v9[v8];
    v11 = v9;
    if ( &v9[v8] == v9 )
    {
      v8 = 0;
      v12 = 0;
    }
    else
    {
      do
        *v9++ = v6;
      while ( v10 != v9 );
      v12 = (v8 * 8) >> 3;
    }
    v2 = sub_159DFD0(v7, v11, v12);
    if ( v11 )
      j_j___libc_free_0(v11, v8 * 8);
    return v2;
  }
  return sub_159DFD0(v7, 0, 0);
}
