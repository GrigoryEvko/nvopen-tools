// Function: sub_1BE6790
// Address: 0x1be6790
//
__int64 __fastcall sub_1BE6790(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v5; // rax
  __int64 *v6; // r12
  int v7; // ebx
  unsigned __int64 v8; // r13
  __int64 v9; // rax
  _QWORD *v10; // r14
  unsigned int i; // r15d
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v20; // [rsp+10h] [rbp-40h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v21 = sub_157F1C0(a2);
  result = sub_1BE63B0(a1, v21, a2);
  if ( a3 != v21 )
  {
    while ( 1 )
    {
      v5 = sub_157EBA0(v21);
      v6 = (__int64 *)v5;
      if ( !v5 )
        break;
      v7 = sub_15F4D60(v5);
      v8 = sub_157EBA0(v21);
      if ( (unsigned __int64)v7 > 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v20 = 8LL * v7;
      if ( !v7 )
      {
        v6 = 0;
        goto LABEL_17;
      }
      v9 = sub_22077B0(8LL * v7);
      v6 = (__int64 *)v9;
      v10 = (_QWORD *)v9;
      for ( i = 0; i != v7; ++i )
      {
        v12 = sub_15F4DF0(v8, i);
        if ( v10 )
          *v10 = v12;
        ++v10;
      }
      v13 = *v6;
      if ( v7 != 1 )
        goto LABEL_11;
      v15 = v21;
      v17 = a1;
      v16 = *v6;
LABEL_14:
      sub_1BE63B0(v17, v16, v15);
      result = j_j___libc_free_0(v6, v20);
      v21 = v13;
      if ( v13 == a3 )
        return result;
    }
    v20 = 0;
LABEL_17:
    v13 = MEMORY[0];
LABEL_11:
    v14 = v6[1];
    if ( v14 == sub_157F1C0(v13) )
    {
      v13 = v6[1];
      v14 = *v6;
    }
    sub_1BE63B0(a1, v14, v21);
    v15 = v21;
    v16 = v13;
    v17 = a1;
    goto LABEL_14;
  }
  return result;
}
