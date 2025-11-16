// Function: sub_1B44750
// Address: 0x1b44750
//
__int64 __fastcall sub_1B44750(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 **v6; // r8
  char v7; // al
  __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r12
  _BYTE *v14; // rdx
  int v15; // ecx
  _BYTE *v16; // r9
  __int64 *v17; // rax
  __int64 v18; // rcx
  unsigned int v19; // ebx
  __int64 **v21; // [rsp+0h] [rbp-70h]
  __int64 v22; // [rsp+8h] [rbp-68h]
  _BYTE *v23; // [rsp+10h] [rbp-60h] BYREF
  __int64 v24; // [rsp+18h] [rbp-58h]
  _BYTE v25[80]; // [rsp+20h] [rbp-50h] BYREF

  v6 = a2;
  if ( *(_BYTE *)(a1 + 16) > 0x17u )
  {
    v7 = sub_1C30710(a1);
    v6 = a2;
    if ( v7 )
      return 0xFFFFFFFFLL;
  }
  v8 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v9 = *(__int64 **)(a1 - 8);
    v10 = (__int64)&v9[v8];
  }
  else
  {
    v9 = (__int64 *)(a1 - v8 * 8);
    v10 = a1;
  }
  v11 = v10 - (_QWORD)v9;
  v23 = v25;
  v24 = 0x400000000LL;
  v12 = 0xAAAAAAAAAAAAAAABLL * (v11 >> 3);
  v13 = v12;
  if ( (unsigned __int64)v11 > 0x60 )
  {
    v21 = v6;
    v22 = v11;
    sub_16CD150((__int64)&v23, v25, 0xAAAAAAAAAAAAAAABLL * (v11 >> 3), 8, (int)v6, a6);
    v16 = v23;
    v15 = v24;
    v11 = v22;
    v6 = v21;
    v14 = &v23[8 * (unsigned int)v24];
  }
  else
  {
    v14 = v25;
    v15 = 0;
    v16 = v25;
  }
  if ( v11 > 0 )
  {
    v17 = v9;
    do
    {
      v18 = *v17;
      v14 += 8;
      v17 += 3;
      *((_QWORD *)v14 - 1) = v18;
      --v13;
    }
    while ( v13 );
    v16 = v23;
    v15 = v24;
  }
  LODWORD(v24) = v15 + v12;
  v19 = sub_14A5330(v6, a1, (__int64)v16, (unsigned int)(v15 + v12));
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
  if ( v19 == 4 )
    return 0xFFFFFFFFLL;
  else
    return v19;
}
