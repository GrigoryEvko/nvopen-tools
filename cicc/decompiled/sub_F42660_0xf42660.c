// Function: sub_F42660
// Address: 0xf42660
//
__int64 __fastcall sub_F42660(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // r11d
  int v9; // r9d
  unsigned int i; // eax
  __int64 v11; // r10
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  int v16; // ebx
  unsigned int j; // eax
  __int64 v18; // r10
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rsi
  __int64 v28; // [rsp+10h] [rbp-90h] BYREF
  int *v29; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+20h] [rbp-80h]
  __int64 v31; // [rsp+28h] [rbp-78h]
  int v32; // [rsp+30h] [rbp-70h] BYREF
  char v33; // [rsp+34h] [rbp-6Ch]
  __int64 v34; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v35; // [rsp+48h] [rbp-58h]
  __int64 v36; // [rsp+50h] [rbp-50h]
  int v37; // [rsp+58h] [rbp-48h]
  char v38; // [rsp+5Ch] [rbp-44h]
  _BYTE v39[64]; // [rsp+60h] [rbp-40h] BYREF

  v6 = *(unsigned int *)(a4 + 88);
  v7 = *(_QWORD *)(a4 + 72);
  if ( (_DWORD)v6 )
  {
    v8 = 1;
    v9 = v6 - 1;
    for ( i = (v6 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v9 & v12 )
    {
      v11 = v7 + 24LL * i;
      if ( *(_UNKNOWN **)v11 == &unk_4F81450 && a3 == *(_QWORD *)(v11 + 8) )
        break;
      if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
        goto LABEL_7;
      v12 = v8 + i;
      ++v8;
    }
    v15 = v7 + 24 * v6;
    if ( v15 != v11 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL);
      if ( v13 )
        v13 += 8;
      goto LABEL_14;
    }
  }
  else
  {
LABEL_7:
    v11 = v7 + 24LL * (unsigned int)v6;
    if ( !(_DWORD)v6 )
    {
      v13 = 0;
LABEL_9:
      v14 = 0;
      goto LABEL_22;
    }
    v9 = v6 - 1;
  }
  v15 = v11;
  v13 = 0;
LABEL_14:
  v16 = 1;
  for ( j = v9
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v9 & v19 )
  {
    v18 = v7 + 24LL * j;
    if ( *(_UNKNOWN **)v18 == &unk_4F875F0 && a3 == *(_QWORD *)(v18 + 8) )
      break;
    if ( *(_QWORD *)v18 == -4096 && *(_QWORD *)(v18 + 8) == -4096 )
      goto LABEL_9;
    v19 = v16 + j;
    ++v16;
  }
  if ( v18 == v15 )
    goto LABEL_9;
  v14 = *(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL);
  if ( v14 )
    v14 += 8;
LABEL_22:
  v28 = v13;
  v30 = v14;
  v29 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 1;
  if ( (unsigned int)sub_F34EF0(a3, (__int64)&v28) )
  {
    BYTE4(v31) = 1;
    v35 = v39;
    v29 = &v32;
    v28 = 0;
    v30 = 2;
    LODWORD(v31) = 0;
    v34 = 0;
    v36 = 2;
    v37 = 0;
    v38 = 1;
    sub_F42230((__int64)&v28, (__int64)&unk_4F81450, v20, (__int64)&v32, v21, (__int64)v39);
    sub_F42230((__int64)&v28, (__int64)&unk_4F875F0, v23, v24, v25, v26);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v32, (__int64)&v28);
    v27 = a1 + 80;
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v39, (__int64)&v34);
    if ( !v38 )
      _libc_free(v35, v27);
    if ( !BYTE4(v31) )
      _libc_free(v29, v27);
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
    *(_QWORD *)(a1 + 32) = &unk_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
