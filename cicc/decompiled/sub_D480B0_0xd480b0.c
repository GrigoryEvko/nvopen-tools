// Function: sub_D480B0
// Address: 0xd480b0
//
__int64 __fastcall sub_D480B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // r8
  __int64 v5; // rdx
  _BYTE *i; // rax
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  _BYTE *v12; // rcx
  __int64 v13; // r15
  const void *v14; // r13
  size_t v15; // rbx
  __int64 v16; // r14
  __int64 *v18; // [rsp+8h] [rbp-88h]
  __int64 *v19; // [rsp+10h] [rbp-80h]
  __int64 v20; // [rsp+28h] [rbp-68h]
  __int64 v21; // [rsp+28h] [rbp-68h]
  _BYTE *v22; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+38h] [rbp-58h]
  _BYTE v24[80]; // [rsp+40h] [rbp-50h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v2 = *(__int64 **)(a2 + 32);
  v23 = 0x400000000LL;
  v3 = *(__int64 **)(a2 + 40);
  v22 = v24;
  v18 = v3;
  if ( v2 != v3 )
  {
    v4 = *v2;
    v5 = 0;
    v19 = v2 + 1;
    for ( i = v24; ; i = v22 )
    {
      *(_QWORD *)&i[8 * v5] = v4;
      LODWORD(v5) = v23 + 1;
      LODWORD(v23) = v23 + 1;
      do
      {
        v12 = v22;
        v7 = HIDWORD(v23);
        v13 = *(_QWORD *)&v22[8 * (unsigned int)v5 - 8];
        v9 = (unsigned int)(v5 - 1);
        LODWORD(v23) = v5 - 1;
        v8 = *(_QWORD *)(v13 + 16);
        v14 = *(const void **)(v13 + 8);
        v15 = v8 - (_QWORD)v14;
        v16 = (v8 - (__int64)v14) >> 3;
        if ( v16 + v9 > (unsigned __int64)HIDWORD(v23) )
        {
          v7 = (__int64)v24;
          v20 = *(_QWORD *)(v13 + 16);
          sub_C8D5F0((__int64)&v22, v24, v16 + v9, 8u, v4, v8);
          v12 = v22;
          v9 = (unsigned int)v23;
          v8 = v20;
        }
        if ( v14 != (const void *)v8 )
        {
          v7 = (__int64)v14;
          memmove(&v12[8 * v9], v14, v15);
          LODWORD(v9) = v23;
        }
        v10 = *(unsigned int *)(a1 + 12);
        LODWORD(v23) = v16 + v9;
        v11 = *(unsigned int *)(a1 + 8);
        if ( v11 + 1 > v10 )
        {
          v7 = a1 + 16;
          sub_C8D5F0(a1, (const void *)(a1 + 16), v11 + 1, 8u, v4, v8);
          v11 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v11) = v13;
        v5 = (unsigned int)v23;
        ++*(_DWORD *)(a1 + 8);
      }
      while ( (_DWORD)v5 );
      if ( v18 == v19 )
        break;
      v4 = *v19;
      if ( !HIDWORD(v23) )
      {
        v21 = *v19;
        sub_C8D5F0((__int64)&v22, v24, 1u, 8u, v4, v8);
        v5 = (unsigned int)v23;
        v4 = v21;
      }
      ++v19;
    }
    if ( v22 != v24 )
      _libc_free(v22, v7);
  }
  return a1;
}
