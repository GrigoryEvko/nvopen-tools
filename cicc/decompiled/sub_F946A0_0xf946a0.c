// Function: sub_F946A0
// Address: 0xf946a0
//
__int64 __fastcall sub_F946A0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 **v3; // r9
  int v4; // r8d
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // r12
  __int64 v9; // r10
  __int64 v10; // rbx
  unsigned __int8 **v11; // rcx
  int v12; // eax
  unsigned __int8 **v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // r12
  unsigned __int8 **v18; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+28h] [rbp-58h]
  _BYTE v20[80]; // [rsp+30h] [rbp-50h] BYREF

  v3 = (__int64 **)a1;
  v4 = a3;
  v5 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(_QWORD *)(a2 - 8);
    v7 = v6 + v5;
  }
  else
  {
    v6 = a2 - v5;
    v7 = a2;
  }
  v8 = v7 - v6;
  v18 = (unsigned __int8 **)v20;
  v9 = v8 >> 5;
  v19 = 0x400000000LL;
  v10 = v8 >> 5;
  if ( (unsigned __int64)v8 > 0x80 )
  {
    sub_C8D5F0((__int64)&v18, v20, v8 >> 5, 8u, a3, a1);
    v13 = v18;
    v12 = v19;
    v9 = v8 >> 5;
    v3 = (__int64 **)a1;
    v4 = a3;
    v11 = &v18[(unsigned int)v19];
  }
  else
  {
    v11 = (unsigned __int8 **)v20;
    v12 = 0;
    v13 = (unsigned __int8 **)v20;
  }
  if ( v8 > 0 )
  {
    v14 = 0;
    do
    {
      v11[v14 / 8] = *(unsigned __int8 **)(v6 + 4 * v14);
      v14 += 8LL;
      --v10;
    }
    while ( v10 );
    v13 = v18;
    v12 = v19;
  }
  LODWORD(v19) = v9 + v12;
  v15 = sub_DFCEF0(v3, (unsigned __int8 *)a2, v13, (unsigned int)(v9 + v12), v4);
  if ( v18 != (unsigned __int8 **)v20 )
    _libc_free(v18, a2);
  return v15;
}
