// Function: sub_28E5740
// Address: 0x28e5740
//
__int64 __fastcall sub_28E5740(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // r15
  int v12; // r10d
  unsigned int i; // eax
  __int64 v14; // rdi
  unsigned int v15; // eax
  unsigned __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  void *v19; // rsi
  unsigned __int64 v21; // [rsp+8h] [rbp-98h]
  __int64 v22; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v23; // [rsp+18h] [rbp-88h]
  int v24; // [rsp+20h] [rbp-80h]
  int v25; // [rsp+24h] [rbp-7Ch]
  int v26; // [rsp+28h] [rbp-78h]
  char v27; // [rsp+2Ch] [rbp-74h]
  _QWORD v28[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v29; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v30; // [rsp+48h] [rbp-58h]
  __int64 v31; // [rsp+50h] [rbp-50h]
  int v32; // [rsp+58h] [rbp-48h]
  char v33; // [rsp+5Ch] [rbp-44h]
  _BYTE v34[64]; // [rsp+60h] [rbp-40h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v8 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  v11 = v8 + 8;
  if ( !(_DWORD)v9 )
    goto LABEL_18;
  v12 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v15 )
  {
    v14 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v14 == &unk_4F81450 && a3 == *(_QWORD *)(v14 + 8) )
      break;
    if ( *(_QWORD *)v14 == -4096 && *(_QWORD *)(v14 + 8) == -4096 )
      goto LABEL_18;
    v15 = v12 + i;
    ++v12;
  }
  if ( v14 == v10 + 24 * v9 )
  {
LABEL_18:
    v16 = 0;
  }
  else
  {
    v16 = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL);
    if ( v16 )
      v16 += 8LL;
  }
  v21 = v16;
  sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v19 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_28E4750(a3, v7, v11, v21, v17, v18) )
  {
    v23 = v28;
    v24 = 2;
    v28[0] = &unk_4F81450;
    v26 = 0;
    v27 = 1;
    v29 = 0;
    v30 = v34;
    v31 = 2;
    v32 = 0;
    v33 = 1;
    v25 = 1;
    v22 = 1;
    sub_C8CF70(a1, v19, 2, (__int64)v28, (__int64)&v22);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v34, (__int64)&v29);
    if ( !v33 )
      _libc_free((unsigned __int64)v30);
    if ( !v27 )
      _libc_free((unsigned __int64)v23);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v19;
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
