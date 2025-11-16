// Function: sub_1054920
// Address: 0x1054920
//
__int64 __fastcall sub_1054920(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v8; // rbx
  __int64 i; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rsi
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-5F8h]
  _BYTE *v20; // [rsp+20h] [rbp-5E0h] BYREF
  __int64 v21; // [rsp+28h] [rbp-5D8h]
  _BYTE v22[64]; // [rsp+30h] [rbp-5D0h] BYREF
  __int64 v23[178]; // [rsp+70h] [rbp-590h] BYREF

  v6 = a3 + 72;
  v8 = *(_QWORD *)(a3 + 80);
  v20 = v22;
  v21 = 0x800000000LL;
  if ( a3 + 72 == v8 )
  {
    i = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v8 + 32);
      if ( i != v8 + 24 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v6 == v8 )
        goto LABEL_7;
      if ( !v8 )
        BUG();
    }
  }
  while ( v6 != v8 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 60 )
    {
      v16 = (unsigned int)v21;
      v17 = (unsigned int)v21 + 1LL;
      if ( v17 > HIDWORD(v21) )
      {
        v19 = v6;
        sub_C8D5F0((__int64)&v20, v22, v17, 8u, v6, a6);
        v16 = (unsigned int)v21;
        v6 = v19;
      }
      *(_QWORD *)&v20[8 * v16] = i - 24;
      LODWORD(v21) = v21 + 1;
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v8 + 32) )
    {
      v18 = v8 - 24;
      if ( !v8 )
        v18 = 0;
      if ( i != v18 + 48 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v6 == v8 )
        goto LABEL_7;
      if ( !v8 )
        BUG();
    }
  }
LABEL_7:
  sub_1054520((__int64)v23, a3, (__int64)v20, (unsigned int)v21, *a2, a6);
  sub_1051870((__int64)v23, a3, v10, v11, v12, v13);
  v14 = *((_QWORD *)a2 + 1);
  sub_104DD10(v23, v14);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)a1 = 1;
  sub_D896C0((__int64)v23, v14);
  if ( v20 != v22 )
    _libc_free(v20, v14);
  return a1;
}
