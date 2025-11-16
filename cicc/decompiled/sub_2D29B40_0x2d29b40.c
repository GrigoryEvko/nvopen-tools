// Function: sub_2D29B40
// Address: 0x2d29b40
//
void __fastcall sub_2D29B40(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rbx
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r9
  unsigned int v6; // edx
  unsigned __int64 v7; // rbx
  __int64 v8; // rsi
  unsigned int *v9; // r15
  unsigned __int64 v10; // r13
  unsigned int v11; // ebx
  unsigned int v12; // ebx
  unsigned __int64 v13[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2;
  v3 = a1[2];
  v4 = *(_QWORD *)a1;
  v5 = v3 + 1;
  v6 = a1[2];
  if ( v3 + 1 > (unsigned __int64)a1[3] )
  {
    v9 = a1 + 4;
    if ( v4 > a2 || a2 >= v4 + 32 * v3 )
    {
      v4 = sub_C8D7D0((__int64)a1, (__int64)(a1 + 4), v5, 0x20u, v13, v5);
      sub_2D296B0(a1, v4);
      v12 = v13[0];
      if ( *(unsigned int **)a1 != v9 )
        _libc_free(*(_QWORD *)a1);
      a1[3] = v12;
      v3 = a1[2];
      *(_QWORD *)a1 = v4;
      v6 = v3;
    }
    else
    {
      v10 = a2 - v4;
      v4 = sub_C8D7D0((__int64)a1, (__int64)(a1 + 4), v5, 0x20u, v13, v5);
      sub_2D296B0(a1, v4);
      v11 = v13[0];
      if ( *(unsigned int **)a1 != v9 )
        _libc_free(*(_QWORD *)a1);
      a1[3] = v11;
      v3 = a1[2];
      v2 = v4 + v10;
      *(_QWORD *)a1 = v4;
      v6 = v3;
    }
  }
  v7 = v4 + 32 * v3;
  if ( v7 )
  {
    *(_DWORD *)v7 = *(_DWORD *)v2;
    *(_QWORD *)(v7 + 8) = *(_QWORD *)(v2 + 8);
    v8 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v7 + 16) = v8;
    if ( v8 )
      sub_B96E90(v7 + 16, v8, 1);
    *(_QWORD *)(v7 + 24) = *(_QWORD *)(v2 + 24);
    v6 = a1[2];
  }
  a1[2] = v6 + 1;
}
