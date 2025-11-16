// Function: sub_2774890
// Address: 0x2774890
//
__int64 __fastcall sub_2774890(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  _QWORD *v7; // r12
  _QWORD *v8; // r14
  char v9; // dl
  __int64 v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // ebx
  __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v17 = sub_C8D7D0(a1, a1 + 16, a2, 0x40u, v18, a6);
  v6 = v17;
  v7 = (_QWORD *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
  if ( *(_QWORD **)a1 != v7 )
  {
    v8 = *(_QWORD **)a1;
    do
    {
      if ( v6 )
      {
        v9 = *(_BYTE *)v8;
        *(_QWORD *)(v6 + 8) = 0;
        *(_QWORD *)(v6 + 16) = 0;
        *(_BYTE *)v6 = v9;
        v10 = v8[3];
        *(_QWORD *)(v6 + 24) = v10;
        if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
          sub_BD6050((unsigned __int64 *)(v6 + 8), v8[1] & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v6 + 32) = 0;
        *(_QWORD *)(v6 + 40) = 0;
        v11 = v8[6];
        *(_QWORD *)(v6 + 48) = v11;
        if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
          sub_BD6050((unsigned __int64 *)(v6 + 32), v8[4] & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v6 + 56) = v8[7];
      }
      v8 += 8;
      v6 += 64;
    }
    while ( v7 != v8 );
    v12 = *(_QWORD **)a1;
    v7 = (_QWORD *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
    if ( *(_QWORD **)a1 != v7 )
    {
      do
      {
        v13 = *(v7 - 2);
        v7 -= 8;
        if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
          sub_BD60C0(v7 + 4);
        v14 = v7[3];
        if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
          sub_BD60C0(v7 + 1);
      }
      while ( v7 != v12 );
      v7 = *(_QWORD **)a1;
    }
  }
  v15 = v18[0];
  if ( (_QWORD *)(a1 + 16) != v7 )
    _libc_free((unsigned __int64)v7);
  *(_DWORD *)(a1 + 12) = v15;
  *(_QWORD *)a1 = v17;
  return v17;
}
