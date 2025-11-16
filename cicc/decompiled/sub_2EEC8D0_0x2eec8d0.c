// Function: sub_2EEC8D0
// Address: 0x2eec8d0
//
void __fastcall sub_2EEC8D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int v11; // ebx
  void *v12; // rax
  int v13; // eax
  __int64 v14; // r12
  __int64 v15; // r14
  bool v16; // zf
  __int64 v17; // rbx
  __int64 i; // r12
  unsigned __int64 v19; // rdi
  _BYTE *v20; // [rsp+10h] [rbp-160h] BYREF
  __int64 v21; // [rsp+18h] [rbp-158h]
  _BYTE v22[64]; // [rsp+20h] [rbp-150h] BYREF
  unsigned __int64 v23[2]; // [rsp+60h] [rbp-110h] BYREF
  _BYTE v24[192]; // [rsp+70h] [rbp-100h] BYREF
  unsigned __int64 v25; // [rsp+130h] [rbp-40h]
  unsigned int v26; // [rsp+138h] [rbp-38h]

  v20 = v22;
  v21 = 0x800000000LL;
  do
  {
    v7 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(a2 + 24);
    if ( *(_BYTE *)(v7 + 32) )
      break;
    v8 = (unsigned int)v21;
    v9 = (unsigned int)v21 + 1LL;
    if ( v9 > HIDWORD(v21) )
    {
      sub_C8D5F0((__int64)&v20, v22, v9, 8u, a5, a6);
      v8 = (unsigned int)v21;
    }
    *(_QWORD *)&v20[8 * v8] = a2;
    LODWORD(v21) = v21 + 1;
    a2 = *(_QWORD *)v7;
  }
  while ( *(_QWORD *)v7 );
  v25 = 0;
  v23[0] = (unsigned __int64)v24;
  v23[1] = 0x800000000LL;
  v10 = *(_QWORD *)(a1 + 440);
  v26 = 0;
  v11 = *(_DWORD *)(*(_QWORD *)(v10 + 16) + 44LL);
  if ( v11 )
  {
    v12 = _libc_calloc(v11, 1u);
    if ( !v12 )
      sub_C64F00("Allocation failed", 1u);
    v25 = (unsigned __int64)v12;
    v26 = v11;
  }
  v13 = v21;
  if ( (_DWORD)v21 )
  {
    do
    {
      v14 = *(_QWORD *)&v20[8 * v13 - 8];
      LODWORD(v21) = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(v14 + 24);
      v16 = *(_BYTE *)(v15 + 33) == 0;
      *(_BYTE *)(v15 + 32) = 1;
      *(_DWORD *)(v15 + 36) = 0;
      if ( !v16 )
        *(_DWORD *)(v15 + 36) = sub_2EEC580(a1, v15);
      v17 = *(_QWORD *)(v14 + 56);
      for ( i = v14 + 48; i != v17; v17 = *(_QWORD *)(v17 + 8) )
      {
        while ( 1 )
        {
          sub_2EEB960(a1, v15, v17, (__int64)v23);
          if ( !v17 )
            BUG();
          if ( (*(_BYTE *)v17 & 4) == 0 )
            break;
          v17 = *(_QWORD *)(v17 + 8);
          if ( i == v17 )
            goto LABEL_18;
        }
        while ( (*(_BYTE *)(v17 + 44) & 8) != 0 )
          v17 = *(_QWORD *)(v17 + 8);
      }
LABEL_18:
      v13 = v21;
    }
    while ( (_DWORD)v21 );
    v19 = v25;
    if ( !v25 )
      goto LABEL_21;
  }
  else
  {
    v19 = v25;
    if ( !v25 )
      goto LABEL_23;
  }
  _libc_free(v19);
LABEL_21:
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0]);
LABEL_23:
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
}
