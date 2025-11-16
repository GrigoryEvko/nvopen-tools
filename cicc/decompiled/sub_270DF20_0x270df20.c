// Function: sub_270DF20
// Address: 0x270df20
//
__int64 __fastcall sub_270DF20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r9
  __int64 v5; // r12
  unsigned int v8; // eax
  __int64 v9; // rdx
  unsigned int v10; // r12d
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v18; // [rsp+18h] [rbp-88h]
  int v19; // [rsp+20h] [rbp-80h]
  int v20; // [rsp+24h] [rbp-7Ch]
  int v21; // [rsp+28h] [rbp-78h]
  char v22; // [rsp+2Ch] [rbp-74h]
  _QWORD v23[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v24; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v25; // [rsp+48h] [rbp-58h]
  __int64 v26; // [rsp+50h] [rbp-50h]
  int v27; // [rsp+58h] [rbp-48h]
  char v28; // [rsp+5Ch] [rbp-44h]
  _BYTE v29[64]; // [rsp+60h] [rbp-40h] BYREF

  if ( !byte_5031DC8[0] )
    goto LABEL_2;
  LOBYTE(v8) = sub_270A460(*(_QWORD *)(a3 + 40));
  v10 = v8;
  if ( !(_BYTE)v8 )
    goto LABEL_2;
  v11 = a3 + 72;
  v12 = *(_QWORD *)(a3 + 80);
  if ( v11 == v12 )
    goto LABEL_2;
  if ( !v12 )
    BUG();
  while ( 1 )
  {
    v13 = *(_QWORD *)(v12 + 32);
    if ( v13 != v12 + 24 )
      break;
    v12 = *(_QWORD *)(v12 + 8);
    if ( v11 == v12 )
      goto LABEL_2;
    if ( !v12 )
      BUG();
  }
  if ( v12 == v11 )
  {
LABEL_2:
    v4 = a1 + 32;
    v5 = a1 + 80;
LABEL_3:
    *(_QWORD *)(a1 + 8) = v4;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v5;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v14 = 0;
  do
  {
    if ( !v13 )
      BUG();
    if ( *(_BYTE *)(v13 - 24) == 85 )
    {
      v15 = *(_QWORD *)(v13 - 56);
      if ( v15 )
      {
        if ( !*(_BYTE *)v15 && *(_QWORD *)(v15 + 24) == *(_QWORD *)(v13 + 56) )
        {
          v16 = sub_3108960(v15, v14, v9);
          v14 = (unsigned __int8)v14;
          if ( v16 <= 0xB && ((1LL << v16) & 0xC63) != 0 )
          {
            sub_BD84D0(v13 - 24, *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 - 20) & 0x7FFFFFF) - 24));
            v14 = v10;
          }
        }
      }
    }
    v13 = *(_QWORD *)(v13 + 8);
    v9 = 0;
    while ( v13 == v12 - 24 + 48 )
    {
      v12 = *(_QWORD *)(v12 + 8);
      if ( v11 == v12 )
        goto LABEL_22;
      if ( !v12 )
        BUG();
      v13 = *(_QWORD *)(v12 + 32);
    }
  }
  while ( v11 != v12 );
LABEL_22:
  v4 = a1 + 32;
  v5 = a1 + 80;
  if ( !(_BYTE)v14 )
    goto LABEL_3;
  v18 = v23;
  v19 = 2;
  v23[0] = &unk_4F82408;
  v21 = 0;
  v22 = 1;
  v24 = 0;
  v25 = v29;
  v26 = 2;
  v27 = 0;
  v28 = 1;
  v20 = 1;
  v17 = 1;
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v23, (__int64)&v17);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v29, (__int64)&v24);
  if ( !v28 )
    _libc_free((unsigned __int64)v25);
  if ( !v22 )
    _libc_free((unsigned __int64)v18);
  return a1;
}
