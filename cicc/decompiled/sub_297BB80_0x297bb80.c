// Function: sub_297BB80
// Address: 0x297bb80
//
__int64 __fastcall sub_297BB80(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 v8; // rdx
  _QWORD *v9; // rbx
  __int64 v10; // r15
  _QWORD *v11; // rax
  int v12; // r8d
  _BYTE *v13; // rcx
  int v14; // eax
  __int64 v15; // rsi
  unsigned int v16; // r12d
  __int64 v17; // rax
  int v18; // edx
  _BYTE *v20; // [rsp+0h] [rbp-60h] BYREF
  __int64 v21; // [rsp+8h] [rbp-58h]
  _BYTE v22[80]; // [rsp+10h] [rbp-50h] BYREF

  v7 = *(_DWORD *)(a1 + 4);
  v20 = v22;
  v21 = 0x400000000LL;
  v8 = v7 & 0x7FFFFFF;
  v9 = (_QWORD *)(a1 + 32 * (1 - v8));
  v10 = (-32 * (1 - v8)) >> 5;
  if ( (unsigned __int64)(-32 * (1 - v8)) > 0x80 )
  {
    sub_C8D5F0((__int64)&v20, v22, (-32 * (1 - v8)) >> 5, 8u, a5, a6);
    v13 = v20;
    v12 = v21;
    v11 = &v20[8 * (unsigned int)v21];
  }
  else
  {
    v11 = v22;
    v12 = 0;
    v13 = v22;
  }
  if ( (_QWORD *)a1 != v9 )
  {
    do
    {
      if ( v11 )
        *v11 = *v9;
      v9 += 4;
      ++v11;
    }
    while ( (_QWORD *)a1 != v9 );
    v13 = v20;
    v12 = v21;
  }
  v14 = *(_DWORD *)(a1 + 4);
  v15 = *(_QWORD *)(a1 + 72);
  LODWORD(v21) = v10 + v12;
  v16 = 0;
  v17 = sub_DF9500(a2, v15, *(_QWORD *)(a1 - 32LL * (v14 & 0x7FFFFFF)), (__int64)v13, (unsigned int)(v10 + v12));
  if ( !v18 )
    LOBYTE(v16) = v17 == 0;
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v16;
}
