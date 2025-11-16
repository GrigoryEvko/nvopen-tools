// Function: sub_1269EE0
// Address: 0x1269ee0
//
__int64 __fastcall sub_1269EE0(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  signed __int64 v3; // r14
  void *v4; // r12
  __int64 v5; // rcx
  int v6; // r13d
  unsigned int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r15
  int v11; // ebx
  __int64 v12; // r13
  int v13; // [rsp-68h] [rbp-68h]
  const char *v14; // [rsp-58h] [rbp-58h] BYREF
  char v15; // [rsp-48h] [rbp-48h]
  char v16; // [rsp-47h] [rbp-47h]

  result = a1[54];
  v2 = a1[53];
  if ( v2 != result )
  {
    result -= v2;
    if ( 0xAAAAAAAAAAAAAAABLL * (result >> 3) )
    {
      if ( result < 0 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v3 = 0x5555555555555558LL * (result >> 3);
      v4 = (void *)sub_22077B0(v3);
      memset(v4, 0, v3);
      v5 = a1[53];
      v6 = -1431655765 * ((a1[54] - v5) >> 3);
      if ( v6 )
      {
        v7 = 0;
        while ( 1 )
        {
          v8 = v7++;
          *((_QWORD *)v4 + v8) = sub_15A4A70(*(_QWORD *)(v5 + 24 * v8 + 16), a1[91]);
          if ( v6 == v7 )
            break;
          v5 = a1[53];
        }
      }
      v9 = sub_1645D80(a1[91], v3 >> 3);
      v10 = *a1;
      v11 = v9;
      v16 = 1;
      v13 = sub_159DFD0(v9, v4, v3 >> 3);
      v14 = "llvm.used";
      v15 = 3;
      v12 = sub_1648A60(88, 1);
      if ( v12 )
        sub_15E51E0(v12, v10, v11, 0, 6, v13, (__int64)&v14, 0, 0, 0, 0);
      sub_15E5D20(v12, "llvm.metadata", 13);
      return j_j___libc_free_0(v4, v3);
    }
  }
  return result;
}
