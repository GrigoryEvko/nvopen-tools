// Function: sub_1D5B6E0
// Address: 0x1d5b6e0
//
bool __fastcall sub_1D5B6E0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  int v4; // r9d
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // r14
  _BYTE *v11; // rdx
  int v12; // ecx
  _BYTE *v13; // r9
  __int64 *v14; // rax
  __int64 v15; // rcx
  int v16; // ebx
  __int64 v17; // [rsp-78h] [rbp-78h]
  _BYTE *v18; // [rsp-68h] [rbp-68h] BYREF
  __int64 v19; // [rsp-60h] [rbp-60h]
  _BYTE v20[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    return 0;
  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 || *(_QWORD *)(v2 + 8) || !(unsigned __int8)sub_14AF470(a2, 0, 0, 0) )
    return 0;
  v5 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v6 = *(__int64 **)(a2 - 8);
    v7 = (__int64)&v6[v5];
  }
  else
  {
    v6 = (__int64 *)(a2 - v5 * 8);
    v7 = a2;
  }
  v8 = v7 - (_QWORD)v6;
  v19 = 0x400000000LL;
  v18 = v20;
  v9 = 0xAAAAAAAAAAAAAAABLL * (v8 >> 3);
  v10 = v9;
  if ( (unsigned __int64)v8 > 0x60 )
  {
    v17 = v8;
    sub_16CD150((__int64)&v18, v20, 0xAAAAAAAAAAAAAAABLL * (v8 >> 3), 8, (int)v20, v4);
    v13 = v18;
    v12 = v19;
    v8 = v17;
    v11 = &v18[8 * (unsigned int)v19];
  }
  else
  {
    v11 = v20;
    v12 = 0;
    v13 = v20;
  }
  if ( v8 > 0 )
  {
    v14 = v6;
    do
    {
      v15 = *v14;
      v11 += 8;
      v14 += 3;
      *((_QWORD *)v11 - 1) = v15;
      --v10;
    }
    while ( v10 );
    v13 = v18;
    v12 = v19;
  }
  LODWORD(v19) = v12 + v9;
  v16 = sub_14A5330(a1, a2, (__int64)v13, (unsigned int)(v12 + v9));
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  return v16 > 3;
}
