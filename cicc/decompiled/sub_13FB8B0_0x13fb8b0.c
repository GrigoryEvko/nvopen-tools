// Function: sub_13FB8B0
// Address: 0x13fb8b0
//
__int64 __fastcall sub_13FB8B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rcx
  _BYTE *i; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // r9
  __int64 v14; // r14
  __int64 v15; // rbx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v19; // [rsp+0h] [rbp-90h]
  __int64 v20; // [rsp+10h] [rbp-80h]
  __int64 v21; // [rsp+28h] [rbp-68h]
  _BYTE *v22; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+38h] [rbp-58h]
  _BYTE v24[80]; // [rsp+40h] [rbp-50h] BYREF

  v3 = a1 + 16;
  *(_QWORD *)a1 = v3;
  v19 = v3;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v4 = *(_QWORD *)(a2 + 40);
  v22 = v24;
  v5 = *(_QWORD *)(a2 + 32);
  v23 = 0x400000000LL;
  if ( v5 != v4 )
  {
    v6 = *(_QWORD *)(v4 - 8);
    v7 = 0;
    v20 = v4 - 8;
    for ( i = v24; ; i = v22 )
    {
      *(_QWORD *)&i[8 * v7] = v6;
      LODWORD(v7) = v23 + 1;
      LODWORD(v23) = v23 + 1;
      do
      {
        v9 = (unsigned __int64)v22;
        v10 = (unsigned int)(v7 - 1);
        v11 = *(_QWORD *)&v22[8 * (unsigned int)v7 - 8];
        LODWORD(v23) = v7 - 1;
        v12 = *(_QWORD *)(v11 + 16);
        v13 = v12 - *(_QWORD *)(v11 + 8);
        v14 = v13 >> 3;
        v15 = v13 >> 3;
        if ( v13 >> 3 > (unsigned __int64)HIDWORD(v23) - v10 )
        {
          v21 = *(_QWORD *)(v11 + 16) - *(_QWORD *)(v11 + 8);
          sub_16CD150(&v22, v24, v14 + v10, 8);
          v9 = (unsigned __int64)v22;
          v10 = (unsigned int)v23;
          v13 = v21;
        }
        v16 = v9 + 8 * v10;
        if ( v13 > 0 )
        {
          do
          {
            v16 += 8LL;
            *(_QWORD *)(v16 - 8) = *(_QWORD *)(v12 - 8 * v14 + 8 * v15-- - 8);
          }
          while ( v15 );
          LODWORD(v10) = v23;
        }
        v17 = *(unsigned int *)(a1 + 8);
        LODWORD(v23) = v14 + v10;
        if ( (unsigned int)v17 >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, v19, 0, 8);
          v17 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v17) = v11;
        v7 = (unsigned int)v23;
        ++*(_DWORD *)(a1 + 8);
      }
      while ( (_DWORD)v7 );
      if ( v5 == v20 )
        break;
      v6 = *(_QWORD *)(v20 - 8);
      if ( !HIDWORD(v23) )
      {
        sub_16CD150(&v22, v24, 0, 8);
        v7 = (unsigned int)v23;
      }
      v20 -= 8;
    }
    if ( v22 != v24 )
      _libc_free((unsigned __int64)v22);
  }
  return a1;
}
