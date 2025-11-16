// Function: sub_3145D50
// Address: 0x3145d50
//
__int64 __fastcall sub_3145D50(unsigned int *src, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v6; // r13
  __int64 v7; // rbx
  unsigned int v8; // eax
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  int v11; // esi
  _QWORD *v12; // rdi
  __int64 v13; // r13
  _QWORD *v15; // [rsp+0h] [rbp-60h] BYREF
  __int64 v16; // [rsp+8h] [rbp-58h]
  _QWORD v17[10]; // [rsp+10h] [rbp-50h] BYREF

  v6 = src;
  v7 = src[2];
  v15 = v17;
  v16 = 0x600000001LL;
  v17[0] = v7;
  v8 = v7;
  v9 = (unsigned __int64)(v7 + 63) >> 6;
  if ( v8 > 0x40 )
    v6 = *(unsigned int **)src;
  if ( v9 + 1 > 6 )
  {
    sub_C8D5F0((__int64)&v15, v17, v9 + 1, 8u, a5, a6);
    v11 = v16;
    v12 = v15;
    v10 = (unsigned int)v16;
  }
  else
  {
    v10 = 1;
    v11 = 1;
    v12 = v17;
  }
  if ( 8 * v9 )
  {
    memcpy(&v12[v10], v6, 8 * v9);
    v12 = v15;
    v11 = v16;
  }
  LODWORD(v16) = v11 + v9;
  v13 = sub_CBF760(v12, 8LL * (unsigned int)(v11 + v9));
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  return v13;
}
