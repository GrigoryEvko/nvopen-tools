// Function: sub_C658A0
// Address: 0xc658a0
//
__int64 __fastcall sub_C658A0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 *v6; // rsi
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *i; // r13
  int v14; // ebx
  __int64 v15; // r14
  int v16; // eax
  __int64 result; // rax
  __int64 v18; // [rsp+8h] [rbp-E8h]
  __int64 v19; // [rsp+10h] [rbp-E0h]
  __int64 **v20; // [rsp+20h] [rbp-D0h]
  _BYTE *v21; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+38h] [rbp-B8h]
  _BYTE v23[176]; // [rsp+40h] [rbp-B0h] BYREF

  v3 = a2 + 1;
  v6 = (__int64 *)8;
  v7 = *(_DWORD *)(a1 + 8);
  v18 = *(_QWORD *)a1;
  v8 = _libc_calloc(v3, 8);
  if ( !v8 )
  {
    if ( v3 )
      sub_C64F00("Allocation failed", 1u);
    v8 = sub_C65340(1, 8, v9, v10, v11, v12);
  }
  *(_QWORD *)a1 = v8;
  *(_QWORD *)(v8 + 8LL * a2) = -1;
  v21 = v23;
  *(_DWORD *)(a1 + 8) = a2;
  *(_DWORD *)(a1 + 12) = 0;
  v22 = 0x2000000000LL;
  if ( v7 )
  {
    v20 = (__int64 **)v18;
    v19 = v18 + 8LL * (unsigned int)(v7 - 1) + 8;
    do
    {
      for ( i = *v20; i; LODWORD(v22) = 0 )
      {
        if ( ((unsigned __int8)i & 1) != 0 )
          break;
        v6 = i;
        i = (__int64 *)*i;
        *v6 = 0;
        v14 = *(_DWORD *)(a1 + 8);
        v15 = *(_QWORD *)a1;
        v16 = (*(__int64 (__fastcall **)(__int64, __int64 *, _BYTE **))(a3 + 16))(a1, v6, &v21);
        sub_C657C0((__int64 *)a1, v6, (__int64 *)(v15 + 8LL * (v16 & (unsigned int)(v14 - 1))), a3);
      }
      ++v20;
    }
    while ( v20 != (__int64 **)v19 );
  }
  result = _libc_free(v18, v6);
  if ( v21 != v23 )
    return _libc_free(v21, v6);
  return result;
}
