// Function: sub_2752520
// Address: 0x2752520
//
__int64 __fastcall sub_2752520(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  char v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdi
  char v8; // al
  __int64 v10; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v11; // [rsp+8h] [rbp-88h]
  int v12; // [rsp+10h] [rbp-80h]
  int v13; // [rsp+14h] [rbp-7Ch]
  int v14; // [rsp+18h] [rbp-78h]
  char v15; // [rsp+1Ch] [rbp-74h]
  _QWORD v16[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v17; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v18; // [rsp+38h] [rbp-58h]
  __int64 v19; // [rsp+40h] [rbp-50h]
  int v20; // [rsp+48h] [rbp-48h]
  char v21; // [rsp+4Ch] [rbp-44h]
  _BYTE v22[64]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a3 + 72;
  v5 = 0;
  v6 = *(_QWORD *)(a3 + 80);
  if ( v6 == a3 + 72 )
    goto LABEL_9;
  do
  {
    v7 = v6 - 24;
    if ( !v6 )
      v7 = 0;
    v8 = sub_F3F2F0(v7, a2);
    v6 = *(_QWORD *)(v6 + 8);
    v5 |= v8;
  }
  while ( v4 != v6 );
  if ( !v5 )
  {
LABEL_9:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v11 = v16;
  v12 = 2;
  v16[0] = &unk_4F82408;
  v14 = 0;
  v15 = 1;
  v17 = 0;
  v18 = v22;
  v19 = 2;
  v20 = 0;
  v21 = 1;
  v13 = 1;
  v10 = 1;
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v16, (__int64)&v10);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v22, (__int64)&v17);
  if ( !v21 )
  {
    _libc_free((unsigned __int64)v18);
    if ( v15 )
      return a1;
    goto LABEL_10;
  }
  if ( !v15 )
LABEL_10:
    _libc_free((unsigned __int64)v11);
  return a1;
}
