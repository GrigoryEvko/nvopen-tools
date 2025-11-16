// Function: sub_2903DC0
// Address: 0x2903dc0
//
__int64 __fastcall sub_2903DC0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // rbx
  _QWORD *v7; // r12
  _QWORD *v8; // r14
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // ebx
  __int64 v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v16 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v17, a6);
  v6 = (unsigned __int64 *)v16;
  v7 = (_QWORD *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v7 )
  {
    v8 = *(_QWORD **)a1;
    do
    {
      if ( v6 )
      {
        *v6 = 0;
        v6[1] = 0;
        v9 = v8[2];
        v6[2] = v9;
        if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
          sub_BD6050(v6, *v8 & 0xFFFFFFFFFFFFFFF8LL);
        v6[3] = 0;
        v6[4] = 0;
        v10 = v8[5];
        v6[5] = v10;
        if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
          sub_BD6050(v6 + 3, v8[3] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v8 += 6;
      v6 += 6;
    }
    while ( v7 != v8 );
    v11 = *(_QWORD **)a1;
    v7 = (_QWORD *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v7 )
    {
      do
      {
        v12 = *(v7 - 1);
        v7 -= 6;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD60C0(v7 + 3);
        v13 = v7[2];
        if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
          sub_BD60C0(v7);
      }
      while ( v7 != v11 );
      v7 = *(_QWORD **)a1;
    }
  }
  v14 = v17[0];
  if ( (_QWORD *)(a1 + 16) != v7 )
    _libc_free((unsigned __int64)v7);
  *(_DWORD *)(a1 + 12) = v14;
  *(_QWORD *)a1 = v16;
  return v16;
}
