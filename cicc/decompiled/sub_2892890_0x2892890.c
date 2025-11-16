// Function: sub_2892890
// Address: 0x2892890
//
__int64 __fastcall sub_2892890(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r11d
  unsigned int i; // eax
  __int64 v11; // r8
  unsigned int v12; // eax
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  void *v15; // rsi
  __int64 v17; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-78h]
  int v19; // [rsp+10h] [rbp-70h]
  int v20; // [rsp+14h] [rbp-6Ch]
  int v21; // [rsp+18h] [rbp-68h]
  char v22; // [rsp+1Ch] [rbp-64h]
  _QWORD v23[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v25; // [rsp+38h] [rbp-48h]
  __int64 v26; // [rsp+40h] [rbp-40h]
  int v27; // [rsp+48h] [rbp-38h]
  char v28; // [rsp+4Ch] [rbp-34h]
  _BYTE v29[48]; // [rsp+50h] [rbp-30h] BYREF

  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v7 )
    goto LABEL_18;
  v9 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
  {
    v11 = v8 + 24LL * i;
    if ( *(_UNKNOWN **)v11 == &unk_4F81450 && a3 == *(_QWORD *)(v11 + 8) )
      break;
    if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
      goto LABEL_18;
    v12 = v9 + i;
    ++v9;
  }
  if ( v11 == v8 + 24 * v7 )
  {
LABEL_18:
    v13 = 0;
  }
  else
  {
    v13 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL);
    if ( v13 )
      v13 += 8LL;
  }
  v14 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v15 = (void *)(a1 + 32);
  if ( !(unsigned __int8)sub_2891760(a3, v14 + 8, v13) )
  {
    *(_QWORD *)(a1 + 8) = v15;
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
    return a1;
  }
  v18 = v23;
  v19 = 2;
  v23[0] = &unk_4F81450;
  v21 = 0;
  v22 = 1;
  v24 = 0;
  v25 = v29;
  v26 = 2;
  v27 = 0;
  v28 = 1;
  v20 = 1;
  v17 = 1;
  sub_C8CF70(a1, v15, 2, (__int64)v23, (__int64)&v17);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v29, (__int64)&v24);
  if ( !v28 )
  {
    _libc_free((unsigned __int64)v25);
    if ( v22 )
      return a1;
LABEL_16:
    _libc_free((unsigned __int64)v18);
    return a1;
  }
  if ( !v22 )
    goto LABEL_16;
  return a1;
}
