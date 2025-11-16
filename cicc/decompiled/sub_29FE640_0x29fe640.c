// Function: sub_29FE640
// Address: 0x29fe640
//
__int64 __fastcall sub_29FE640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // r11d
  unsigned int i; // eax
  __int64 v12; // r8
  unsigned int v13; // eax
  __int64 v14; // rdx
  char v15; // al
  void *v16; // rsi
  __int64 v18; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v19; // [rsp+8h] [rbp-88h]
  int v20; // [rsp+10h] [rbp-80h]
  int v21; // [rsp+14h] [rbp-7Ch]
  int v22; // [rsp+18h] [rbp-78h]
  char v23; // [rsp+1Ch] [rbp-74h]
  _QWORD v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v26; // [rsp+38h] [rbp-58h]
  __int64 v27; // [rsp+40h] [rbp-50h]
  int v28; // [rsp+48h] [rbp-48h]
  char v29; // [rsp+4Ch] [rbp-44h]
  _BYTE v30[64]; // [rsp+50h] [rbp-40h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  v9 = v6 + 8;
  if ( !(_DWORD)v7 )
    goto LABEL_18;
  v10 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v13 )
  {
    v12 = v8 + 24LL * i;
    if ( *(_UNKNOWN **)v12 == &unk_4F81450 && a3 == *(_QWORD *)(v12 + 8) )
      break;
    if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == -4096 )
      goto LABEL_18;
    v13 = v10 + i;
    ++v10;
  }
  if ( v12 == v8 + 24 * v7 )
  {
LABEL_18:
    v14 = 0;
  }
  else
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
    if ( v14 )
      v14 += 8;
  }
  v15 = sub_29FD8C0(a3, v9, v14);
  v16 = (void *)(a1 + 32);
  if ( v15 )
  {
    v19 = v24;
    v20 = 2;
    v24[0] = &unk_4F81450;
    v22 = 0;
    v23 = 1;
    v25 = 0;
    v26 = v30;
    v27 = 2;
    v28 = 0;
    v29 = 1;
    v21 = 1;
    v18 = 1;
    sub_C8CF70(a1, v16, 2, (__int64)v24, (__int64)&v18);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v30, (__int64)&v25);
    if ( !v29 )
      _libc_free((unsigned __int64)v26);
    if ( !v23 )
      _libc_free((unsigned __int64)v19);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v16;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
