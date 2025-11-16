// Function: sub_27D7D30
// Address: 0x27d7d30
//
__int64 __fastcall sub_27D7D30(__int64 a1, int *a2, __int64 a3, __int64 a4)
{
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  int v11; // r10d
  unsigned int i; // eax
  __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rbx
  void *v16; // rsi
  __int64 v18; // [rsp+8h] [rbp-98h]
  __int64 v19; // [rsp+10h] [rbp-90h] BYREF
  void **v20; // [rsp+18h] [rbp-88h]
  __int64 v21; // [rsp+20h] [rbp-80h]
  __int64 v22; // [rsp+28h] [rbp-78h]
  void *v23; // [rsp+30h] [rbp-70h] BYREF
  void *v24; // [rsp+38h] [rbp-68h]
  __int64 v25; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v26; // [rsp+48h] [rbp-58h]
  __int64 v27; // [rsp+50h] [rbp-50h]
  int v28; // [rsp+58h] [rbp-48h]
  char v29; // [rsp+5Ch] [rbp-44h]
  _BYTE v30[64]; // [rsp+60h] [rbp-40h] BYREF

  v7 = *a2;
  v8 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  v18 = v8 + 8;
  if ( !(_DWORD)v9 )
    goto LABEL_21;
  v11 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v14 )
  {
    v13 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F81450 && a3 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_21;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v10 + 24 * v9 )
  {
LABEL_21:
    v15 = 0;
  }
  else
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
    if ( v15 )
      v15 += 8;
  }
  v21 = v15;
  LODWORD(v24) = v7;
  v19 = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v20 = 0;
  v22 = v18;
  v23 = 0;
  v16 = (void *)(a1 + 32);
  if ( !(unsigned __int8)sub_27D4E70((__int64)&v19, a3) )
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
    return a1;
  }
  v20 = &v23;
  v21 = 0x100000002LL;
  LODWORD(v22) = 0;
  BYTE4(v22) = 1;
  v25 = 0;
  v26 = v30;
  v27 = 2;
  v28 = 0;
  v29 = 1;
  v23 = &unk_4F82408;
  v19 = 1;
  if ( &unk_4F82408 != (_UNKNOWN *)&qword_4F82400 && &unk_4F82408 != &unk_4F81450 )
  {
    HIDWORD(v21) = 2;
    v19 = 2;
    v24 = &unk_4F81450;
  }
  sub_C8CF70(a1, v16, 2, (__int64)&v23, (__int64)&v19);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v30, (__int64)&v25);
  if ( !v29 )
  {
    _libc_free((unsigned __int64)v26);
    if ( BYTE4(v22) )
      return a1;
LABEL_19:
    _libc_free((unsigned __int64)v20);
    return a1;
  }
  if ( !BYTE4(v22) )
    goto LABEL_19;
  return a1;
}
