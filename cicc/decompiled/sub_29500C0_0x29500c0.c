// Function: sub_29500C0
// Address: 0x29500c0
//
__int64 __fastcall sub_29500C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // r8
  unsigned int v14; // eax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // [rsp+20h] [rbp-90h] BYREF
  _BYTE *v24; // [rsp+28h] [rbp-88h]
  __int64 v25; // [rsp+30h] [rbp-80h]
  int v26; // [rsp+38h] [rbp-78h]
  char v27; // [rsp+3Ch] [rbp-74h]
  _BYTE v28[16]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v29; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v30; // [rsp+58h] [rbp-58h]
  __int64 v31; // [rsp+60h] [rbp-50h]
  int v32; // [rsp+68h] [rbp-48h]
  char v33; // [rsp+6Ch] [rbp-44h]
  _BYTE v34[64]; // [rsp+70h] [rbp-40h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v8 = *(unsigned int *)(a4 + 88);
  v9 = *(_QWORD *)(a4 + 72);
  v10 = v7 + 8;
  if ( !(_DWORD)v8 )
    goto LABEL_18;
  v11 = 1;
  for ( i = (v8 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v8 - 1) & v14 )
  {
    v13 = v9 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F81450 && a3 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_18;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v9 + 24 * v8 )
  {
LABEL_18:
    v15 = 0;
  }
  else
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
    if ( v15 )
      v15 += 8LL;
  }
  if ( (unsigned __int8)sub_294D310(a3, v10, v15) )
  {
    v27 = 1;
    v30 = v34;
    v24 = v28;
    v23 = 0;
    v25 = 2;
    v26 = 0;
    v29 = 0;
    v31 = 2;
    v32 = 0;
    v33 = 1;
    sub_2946420((__int64)&v23, (__int64)&unk_4F89C30, v16, (__int64)v28, v17, a1 + 48);
    sub_2946420((__int64)&v23, (__int64)&unk_4F81450, v19, v20, v21, v22);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v28, (__int64)&v23);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v34, (__int64)&v29);
    if ( !v33 )
      _libc_free((unsigned __int64)v30);
    if ( !v27 )
      _libc_free((unsigned __int64)v24);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
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
