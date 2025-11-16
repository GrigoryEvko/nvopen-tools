// Function: sub_B4E8A0
// Address: 0xb4e8a0
//
__int64 __fastcall sub_B4E8A0(__int64 a1)
{
  _DWORD *v1; // rsi
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  int v5; // ebx
  int v6; // r14d
  _DWORD *v7; // rax
  _DWORD *v8; // rdx
  __int64 v9; // r8
  unsigned __int64 v10; // rax
  int v11; // edi
  _DWORD *v12; // rsi
  int v13; // edx
  __int64 result; // rax
  unsigned __int64 v15; // [rsp+8h] [rbp-78h]
  _DWORD *v16; // [rsp+10h] [rbp-70h] BYREF
  __int64 v17; // [rsp+18h] [rbp-68h]
  _DWORD v18[24]; // [rsp+20h] [rbp-60h] BYREF

  v1 = v18;
  v3 = *(_QWORD *)(a1 - 64);
  v4 = *(int *)(a1 + 80);
  v16 = v18;
  v5 = *(_DWORD *)(*(_QWORD *)(v3 + 8) + 32LL);
  v17 = 0x1000000000LL;
  if ( !v4 )
    goto LABEL_15;
  v6 = v4;
  v7 = v18;
  if ( v4 > 0x10 )
  {
    v15 = v4;
    sub_C8D5F0(&v16, v18, v4, 4);
    v1 = v16;
    v7 = &v16[(unsigned int)v17];
    v8 = &v16[v15];
    if ( v8 == v7 )
      goto LABEL_8;
  }
  else
  {
    v8 = &v18[v4];
    if ( v8 == v18 )
      goto LABEL_8;
  }
  do
  {
    if ( v7 )
      *v7 = 0;
    ++v7;
  }
  while ( v8 != v7 );
  v1 = v16;
LABEL_8:
  LODWORD(v17) = v6;
  v9 = 4LL * (unsigned int)(v6 - 1) + 4;
  v10 = 0;
  do
  {
    while ( 1 )
    {
      v12 = &v1[v10 / 4];
      v13 = *(_DWORD *)(*(_QWORD *)(a1 + 72) + v10);
      if ( v13 != -1 )
        break;
      v10 += 4LL;
      *v12 = -1;
      v1 = v16;
      if ( v9 == v10 )
        goto LABEL_14;
    }
    v11 = v5 + v13;
    if ( v5 <= v13 )
      v11 = v13 - v5;
    v10 += 4LL;
    *v12 = v11;
    v1 = v16;
  }
  while ( v9 != v10 );
LABEL_14:
  v4 = (unsigned int)v17;
LABEL_15:
  sub_B4E7F0(a1, v1, v4);
  result = sub_BD28A0(a1 - 64, a1 - 32);
  if ( v16 != v18 )
    return _libc_free(v16, a1 - 32);
  return result;
}
