// Function: sub_371B7D0
// Address: 0x371b7d0
//
__int64 __fastcall sub_371B7D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int8 **v8; // rbx
  unsigned __int8 **v9; // r12
  __int64 v10; // r8
  unsigned __int8 **v11; // rax
  int v12; // ecx
  unsigned __int8 **v13; // rdx
  __int64 **v14; // rdi
  __int64 v15; // r12
  __int64 v17; // [rsp+8h] [rbp-78h]
  unsigned __int8 **v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  _BYTE v20[96]; // [rsp+20h] [rbp-60h] BYREF

  v6 = *(_QWORD *)(a2 + 16);
  v7 = 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
  {
    v8 = *(unsigned __int8 ***)(v6 - 8);
    v9 = &v8[(unsigned __int64)v7 / 8];
  }
  else
  {
    v9 = *(unsigned __int8 ***)(a2 + 16);
    v8 = (unsigned __int8 **)(v6 - v7);
  }
  v18 = (unsigned __int8 **)v20;
  v10 = v7 >> 5;
  v19 = 0x600000000LL;
  if ( (unsigned __int64)v7 > 0xC0 )
  {
    v17 = v7 >> 5;
    sub_C8D5F0((__int64)&v18, v20, v7 >> 5, 8u, v10, a6);
    v13 = v18;
    v12 = v19;
    LODWORD(v10) = v17;
    v11 = &v18[(unsigned int)v19];
  }
  else
  {
    v11 = (unsigned __int8 **)v20;
    v12 = 0;
    v13 = (unsigned __int8 **)v20;
  }
  if ( v8 != v9 )
  {
    do
    {
      if ( v11 )
        *v11 = *v8;
      v8 += 4;
      ++v11;
    }
    while ( v8 != v9 );
    v13 = v18;
    v12 = v19;
  }
  v14 = *(__int64 ***)(a1 + 8);
  LODWORD(v19) = v10 + v12;
  v15 = sub_DFCEF0(v14, (unsigned __int8 *)v6, v13, (unsigned int)(v10 + v12), 0);
  if ( v18 != (unsigned __int8 **)v20 )
    _libc_free((unsigned __int64)v18);
  return v15;
}
