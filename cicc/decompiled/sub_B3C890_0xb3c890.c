// Function: sub_B3C890
// Address: 0xb3c890
//
__int64 __fastcall sub_B3C890(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v5; // r13
  _DWORD *v6; // rax
  __int64 v7; // r12
  int v8; // edx
  __int64 v9; // rdi
  _DWORD *v10; // r13
  __int64 v11; // rax
  _QWORD *v12; // r14
  _QWORD *v13; // r15
  int v14; // r13d
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  _DWORD *v18; // [rsp+18h] [rbp-48h]
  _QWORD v19[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v17 = a1 + 16;
  v16 = sub_C8D7D0(a1, a1 + 16, a2, 56, v19);
  v5 = v16;
  v6 = *(_DWORD **)a1;
  v7 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    do
    {
      while ( 1 )
      {
        if ( v5 )
        {
          v8 = *v6;
          *(_DWORD *)(v5 + 16) = 0;
          *(_DWORD *)(v5 + 20) = 1;
          *(_DWORD *)v5 = v8;
          *(_QWORD *)(v5 + 8) = v5 + 24;
          if ( v6[4] )
            break;
        }
        v6 += 14;
        v5 += 56;
        if ( (_DWORD *)v7 == v6 )
          goto LABEL_7;
      }
      v3 = (__int64)(v6 + 2);
      v9 = v5 + 8;
      v18 = v6;
      v5 += 56;
      sub_B3BE00(v9, (__int64)(v6 + 2));
      v6 = v18 + 14;
    }
    while ( (_DWORD *)v7 != v18 + 14 );
LABEL_7:
    v10 = *(_DWORD **)a1;
    v7 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v11 = *(unsigned int *)(v7 - 40);
        v12 = *(_QWORD **)(v7 - 48);
        v7 -= 56;
        v11 *= 32;
        v13 = (_QWORD *)((char *)v12 + v11);
        if ( v12 != (_QWORD *)((char *)v12 + v11) )
        {
          do
          {
            v13 -= 4;
            if ( (_QWORD *)*v13 != v13 + 2 )
            {
              v3 = v13[2] + 1LL;
              j_j___libc_free_0(*v13, v3);
            }
          }
          while ( v12 != v13 );
          v12 = *(_QWORD **)(v7 + 8);
        }
        if ( v12 != (_QWORD *)(v7 + 24) )
          _libc_free(v12, v3);
      }
      while ( (_DWORD *)v7 != v10 );
      v7 = *(_QWORD *)a1;
    }
  }
  v14 = v19[0];
  if ( v17 != v7 )
    _libc_free(v7, v3);
  *(_DWORD *)(a1 + 12) = v14;
  *(_QWORD *)a1 = v16;
  return v16;
}
