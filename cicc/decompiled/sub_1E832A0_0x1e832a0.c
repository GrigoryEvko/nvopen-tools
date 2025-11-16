// Function: sub_1E832A0
// Address: 0x1e832a0
//
void __fastcall sub_1E832A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r12d
  void *v11; // rbx
  int i; // eax
  __int64 v13; // r12
  __int64 v14; // r14
  bool v15; // zf
  __int64 v16; // rbx
  __int64 j; // r12
  _BYTE *v18; // [rsp+10h] [rbp-160h] BYREF
  __int64 v19; // [rsp+18h] [rbp-158h]
  _BYTE v20[64]; // [rsp+20h] [rbp-150h] BYREF
  unsigned __int64 v21[2]; // [rsp+60h] [rbp-110h] BYREF
  _BYTE v22[192]; // [rsp+70h] [rbp-100h] BYREF
  unsigned __int64 v23; // [rsp+130h] [rbp-40h]
  unsigned int v24; // [rsp+138h] [rbp-38h]

  v18 = v20;
  v19 = 0x800000000LL;
  do
  {
    v7 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(a2 + 48);
    if ( *(_BYTE *)(v7 + 32) )
      break;
    v8 = (unsigned int)v19;
    if ( (unsigned int)v19 >= HIDWORD(v19) )
    {
      sub_16CD150((__int64)&v18, v20, 0, 8, a5, a6);
      v8 = (unsigned int)v19;
    }
    *(_QWORD *)&v18[8 * v8] = a2;
    LODWORD(v19) = v19 + 1;
    a2 = *(_QWORD *)v7;
  }
  while ( *(_QWORD *)v7 );
  v23 = 0;
  v21[0] = (unsigned __int64)v22;
  v21[1] = 0x800000000LL;
  v9 = *(_QWORD *)(a1 + 440);
  v24 = 0;
  v10 = *(_DWORD *)(*(_QWORD *)(v9 + 248) + 44LL);
  if ( v10 )
  {
    v11 = _libc_calloc(v10, 1u);
    if ( !v11 )
      sub_16BD1C0("Allocation failed", 1u);
    v23 = (unsigned __int64)v11;
    v24 = v10;
  }
  for ( i = v19; (_DWORD)v19; i = v19 )
  {
    v13 = *(_QWORD *)&v18[8 * i - 8];
    LODWORD(v19) = i - 1;
    v14 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(v13 + 48);
    v15 = *(_BYTE *)(v14 + 33) == 0;
    *(_BYTE *)(v14 + 32) = 1;
    *(_DWORD *)(v14 + 36) = 0;
    if ( !v15 )
      *(_DWORD *)(v14 + 36) = sub_1E82F50(a1, v14);
    v16 = *(_QWORD *)(v13 + 32);
    for ( j = v13 + 24; j != v16; v16 = *(_QWORD *)(v16 + 8) )
    {
      while ( 1 )
      {
        sub_1E82360(a1, v14, v16, (__int64)v21, a5, a6);
        if ( !v16 )
          BUG();
        if ( (*(_BYTE *)v16 & 4) == 0 )
          break;
        v16 = *(_QWORD *)(v16 + 8);
        if ( j == v16 )
          goto LABEL_19;
      }
      while ( (*(_BYTE *)(v16 + 46) & 8) != 0 )
        v16 = *(_QWORD *)(v16 + 8);
    }
LABEL_19:
    ;
  }
  _libc_free(v23);
  if ( (_BYTE *)v21[0] != v22 )
    _libc_free(v21[0]);
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
}
