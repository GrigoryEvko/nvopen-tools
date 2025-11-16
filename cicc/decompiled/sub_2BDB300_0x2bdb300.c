// Function: sub_2BDB300
// Address: 0x2bdb300
//
__int64 __fastcall sub_2BDB300(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rcx
  int v11; // ebx
  unsigned int i; // eax
  __int64 v13; // r9
  unsigned int v14; // eax
  __int64 v15; // r8
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rax
  void *v19; // rsi
  __int64 v21; // [rsp+8h] [rbp-C8h]
  __int64 v22; // [rsp+10h] [rbp-C0h]
  __int64 v23; // [rsp+18h] [rbp-B8h]
  __int64 v25; // [rsp+28h] [rbp-A8h]
  __int64 v26; // [rsp+30h] [rbp-A0h]
  __int64 v27; // [rsp+38h] [rbp-98h]
  __int64 v28; // [rsp+40h] [rbp-90h] BYREF
  _QWORD *v29; // [rsp+48h] [rbp-88h]
  int v30; // [rsp+50h] [rbp-80h]
  int v31; // [rsp+54h] [rbp-7Ch]
  int v32; // [rsp+58h] [rbp-78h]
  char v33; // [rsp+5Ch] [rbp-74h]
  _QWORD v34[2]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v35; // [rsp+70h] [rbp-60h] BYREF
  _BYTE *v36; // [rsp+78h] [rbp-58h]
  __int64 v37; // [rsp+80h] [rbp-50h]
  int v38; // [rsp+88h] [rbp-48h]
  char v39; // [rsp+8Ch] [rbp-44h]
  _BYTE v40[64]; // [rsp+90h] [rbp-40h] BYREF

  v25 = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
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
              | ((unsigned __int64)(((unsigned int)&unk_4F6D3F8 >> 9) ^ ((unsigned int)&unk_4F6D3F8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v8 - 1) & v14 )
  {
    v13 = v9 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F6D3F8 && a3 == *(_QWORD *)(v13 + 8) )
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
      v15 += 8;
  }
  v21 = v10;
  v22 = v15;
  v23 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v26 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v27 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v16 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v17 = sub_BC1CD0(a4, &unk_4F86B68, a3) + 8;
  v18 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v19 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2BDB2E0(a2, a3, v25, v21, v22, v23 + 8, a5, v26 + 8, v27 + 8, v16 + 8, v17, v18 + 8) )
  {
    v29 = v34;
    v30 = 2;
    v34[0] = &unk_4F82408;
    v32 = 0;
    v33 = 1;
    v35 = 0;
    v36 = v40;
    v37 = 2;
    v38 = 0;
    v39 = 1;
    v31 = 1;
    v28 = 1;
    sub_C8CF70(a1, v19, 2, (__int64)v34, (__int64)&v28);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v40, (__int64)&v35);
    if ( !v39 )
      _libc_free((unsigned __int64)v36);
    if ( !v33 )
      _libc_free((unsigned __int64)v29);
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
