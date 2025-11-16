// Function: sub_185C700
// Address: 0x185c700
//
__int64 __fastcall sub_185C700(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  __int64 v4; // r13
  unsigned int v5; // r15d
  __int64 v7; // rbx
  __int64 v8; // r13
  char v9; // al
  __int64 v10; // r8
  __int64 *v11; // rax
  __int64 v12; // r8
  unsigned __int64 v13; // rdi
  __int64 *v14; // rax
  char v15; // dl
  char v16; // al
  __int64 *v17; // rsi
  __int64 *v18; // rcx
  __int64 v19; // [rsp+8h] [rbp-A8h]
  __int64 v20; // [rsp+10h] [rbp-A0h] BYREF
  __int64 *v21; // [rsp+18h] [rbp-98h]
  __int64 *v22; // [rsp+20h] [rbp-90h]
  unsigned int v23; // [rsp+28h] [rbp-88h]
  unsigned int v24; // [rsp+2Ch] [rbp-84h]
  int v25; // [rsp+30h] [rbp-80h]
  char v26[120]; // [rsp+38h] [rbp-78h] BYREF

  LOBYTE(v3) = sub_15E4F60(a1);
  if ( (_BYTE)v3 )
    return 0;
  v4 = *(_QWORD *)(a1 + 80);
  if ( *(_QWORD *)(v4 + 8) != a1 + 72 )
    return 0;
  v7 = *(_QWORD *)(v4 + 24);
  v8 = v4 + 16;
  if ( v7 == v8 )
    return 0;
  v5 = v3;
  while ( 1 )
  {
    if ( !v7 )
      BUG();
    v9 = *(_BYTE *)(v7 - 8);
    if ( v9 != 78 )
    {
      if ( v9 == 25 )
        return 1;
      if ( (unsigned __int8)sub_15F3040(v7 - 24) || sub_15F3330(v7 - 24) )
        return 0;
      goto LABEL_12;
    }
    v10 = *(_QWORD *)(v7 - 48);
    if ( *(_BYTE *)(v10 + 16) )
      return v5;
    if ( (*(_BYTE *)(v10 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v10 + 36) - 35) <= 3 )
      goto LABEL_12;
    v19 = *(_QWORD *)(v7 - 48);
    sub_16CCCB0(&v20, (__int64)v26, a2);
    v11 = v21;
    v12 = v19;
    if ( v22 != v21 )
    {
LABEL_18:
      sub_16CCBA0((__int64)&v20, v19);
      v13 = (unsigned __int64)v22;
      v14 = v21;
      v12 = v19;
      if ( !v15 )
        goto LABEL_35;
      goto LABEL_19;
    }
    v17 = &v21[v24];
    if ( v21 != v17 )
      break;
LABEL_31:
    if ( v24 >= v23 )
      goto LABEL_18;
    ++v24;
    *v17 = v19;
    ++v20;
LABEL_19:
    v16 = sub_185C700(v12, &v20);
    v13 = (unsigned __int64)v22;
    if ( !v16 )
    {
      v14 = v21;
LABEL_35:
      if ( (__int64 *)v13 != v14 )
        _libc_free(v13);
      return v5;
    }
    if ( v22 == v21 )
    {
LABEL_12:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v8 == v7 )
        return 0;
    }
    else
    {
      _libc_free((unsigned __int64)v22);
      v7 = *(_QWORD *)(v7 + 8);
      if ( v8 == v7 )
        return 0;
    }
  }
  v18 = 0;
  while ( v19 != *v11 )
  {
    if ( *v11 == -2 )
      v18 = v11;
    if ( v17 == ++v11 )
    {
      if ( !v18 )
        goto LABEL_31;
      *v18 = v19;
      --v25;
      ++v20;
      goto LABEL_19;
    }
  }
  return v5;
}
