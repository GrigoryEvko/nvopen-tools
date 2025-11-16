// Function: sub_F55A40
// Address: 0xf55a40
//
__int64 __fastcall sub_F55A40(__int64 a1)
{
  unsigned __int64 v2; // rcx
  int v3; // eax
  unsigned __int64 v4; // rcx
  bool v5; // cf
  __int64 v6; // rax
  __int64 v7; // r15
  _QWORD *v8; // rsi
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  _QWORD *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r13
  __int64 v20; // rax
  unsigned __int64 v21; // [rsp+8h] [rbp-78h]
  _QWORD v22[2]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v23[96]; // [rsp+20h] [rbp-60h] BYREF

  v2 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a1 + 48 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v2 )
      BUG();
    v3 = *(unsigned __int8 *)(v2 - 24);
    v4 = v2 - 24;
    v5 = (unsigned int)(v3 - 30) < 0xB;
    v6 = 0;
    if ( v5 )
      v6 = v4;
    v7 = v6;
  }
  v8 = v22;
  v9 = 0;
  v10 = 0;
  v22[0] = v23;
  v22[1] = 0x600000000LL;
  sub_F55920(v7, (__int64)v22);
  while ( 1 )
  {
    v11 = *(_QWORD *)(a1 + 56);
    if ( v11 )
      v11 -= 24;
    if ( v7 == v11 )
      break;
    v12 = *(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v12 )
      BUG();
    v13 = (_QWORD *)(v12 - 24);
    if ( *(_QWORD *)(v12 - 8) )
    {
      v14 = *(_QWORD *)(v12 - 16);
      v21 = *(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v14 + 8) == 11 )
        goto LABEL_20;
      v15 = sub_ACADE0((__int64 **)v14);
      sub_BD84D0((__int64)v13, v15);
      v12 = v21;
    }
    v8 = (_QWORD *)*(unsigned __int8 *)(v12 - 24);
    v16 = (unsigned int)((_DWORD)v8 - 39);
    if ( (unsigned int)v16 <= 0x38 && (v17 = 0x100060000000001LL, _bittest64(&v17, v16))
      || *(_BYTE *)(*(_QWORD *)(v12 - 16) + 8LL) == 11 )
    {
LABEL_20:
      v7 = (__int64)v13;
      sub_B44570((__int64)v13);
    }
    else
    {
      if ( (_BYTE)v8 == 85
        && (v20 = *(_QWORD *)(v12 - 56)) != 0
        && !*(_BYTE *)v20
        && *(_QWORD *)(v20 + 24) == *(_QWORD *)(v12 + 56)
        && (*(_BYTE *)(v20 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v20 + 36) - 68) <= 3 )
      {
        v9 = (unsigned int)(v9 + 1);
      }
      else
      {
        v10 = (unsigned int)(v10 + 1);
      }
      sub_B44570((__int64)v13);
      sub_B43D60(v13);
    }
  }
  v18 = v10 | (v9 << 32);
  if ( (_BYTE *)v22[0] != v23 )
    _libc_free(v22[0], v8);
  return v18;
}
