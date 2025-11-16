// Function: sub_28BF6E0
// Address: 0x28bf6e0
//
__int64 __fastcall sub_28BF6E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // r14
  __int64 *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rcx
  int v13; // r11d
  unsigned int i; // eax
  __int64 v15; // r8
  unsigned int v16; // eax
  __int64 v17; // r8
  void *v18; // rsi
  __int64 v20; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v21; // [rsp+8h] [rbp-88h]
  int v22; // [rsp+10h] [rbp-80h]
  int v23; // [rsp+14h] [rbp-7Ch]
  int v24; // [rsp+18h] [rbp-78h]
  char v25; // [rsp+1Ch] [rbp-74h]
  _QWORD v26[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v27; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v28; // [rsp+38h] [rbp-58h]
  __int64 v29; // [rsp+40h] [rbp-50h]
  int v30; // [rsp+48h] [rbp-48h]
  char v31; // [rsp+4Ch] [rbp-44h]
  _BYTE v32[64]; // [rsp+50h] [rbp-40h] BYREF

  v7 = (__int64 *)(sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8);
  v8 = (__int64 *)(sub_BC1CD0(a4, &unk_4F89C30, a3) + 8);
  v9 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v10 = *(unsigned int *)(a4 + 88);
  v11 = *(_QWORD *)(a4 + 72);
  v12 = v9 + 8;
  if ( !(_DWORD)v10 )
    goto LABEL_18;
  v13 = 1;
  for ( i = (v10 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v10 - 1) & v16 )
  {
    v15 = v11 + 24LL * i;
    if ( *(_UNKNOWN **)v15 == &unk_4F81450 && a3 == *(_QWORD *)(v15 + 8) )
      break;
    if ( *(_QWORD *)v15 == -4096 && *(_QWORD *)(v15 + 8) == -4096 )
      goto LABEL_18;
    v16 = v13 + i;
    ++v13;
  }
  if ( v15 == v11 + 24 * v10 )
  {
LABEL_18:
    v17 = 0;
  }
  else
  {
    v17 = *(_QWORD *)(*(_QWORD *)(v15 + 16) + 24LL);
    if ( v17 )
      v17 += 8;
  }
  v18 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_28BF280(a3, v7, v8, v12, v17) )
  {
    v21 = v26;
    v22 = 2;
    v26[0] = &unk_4F81450;
    v24 = 0;
    v25 = 1;
    v27 = 0;
    v28 = v32;
    v29 = 2;
    v30 = 0;
    v31 = 1;
    v23 = 1;
    v20 = 1;
    sub_C8CF70(a1, v18, 2, (__int64)v26, (__int64)&v20);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v32, (__int64)&v27);
    if ( !v31 )
      _libc_free((unsigned __int64)v28);
    if ( !v25 )
      _libc_free((unsigned __int64)v21);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v18;
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
