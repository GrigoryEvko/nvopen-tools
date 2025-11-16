// Function: sub_29D6330
// Address: 0x29d6330
//
__int64 __fastcall sub_29D6330(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r10d
  unsigned int i; // eax
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // r15
  __int64 v14; // rbx
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // [rsp+20h] [rbp-90h] BYREF
  _BYTE *v30; // [rsp+28h] [rbp-88h]
  __int64 v31; // [rsp+30h] [rbp-80h]
  int v32; // [rsp+38h] [rbp-78h]
  char v33; // [rsp+3Ch] [rbp-74h]
  _BYTE v34[16]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v35; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v36; // [rsp+58h] [rbp-58h]
  __int64 v37; // [rsp+60h] [rbp-50h]
  int v38; // [rsp+68h] [rbp-48h]
  char v39; // [rsp+6Ch] [rbp-44h]
  _BYTE v40[64]; // [rsp+70h] [rbp-40h] BYREF

  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v7 )
    goto LABEL_18;
  v9 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
  {
    v11 = v8 + 24LL * i;
    if ( *(_UNKNOWN **)v11 == &unk_4F875F0 && a3 == *(_QWORD *)(v11 + 8) )
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
      v13 += 8;
  }
  v14 = sub_BC1CD0(a4, &unk_4F92388, a3);
  v15 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v16 = sub_29D3E80(v14 + 8, v15 + 8, v13);
  v19 = a1 + 32;
  if ( v16 )
  {
    v36 = v40;
    v30 = v34;
    v29 = 0;
    v31 = 2;
    v32 = 0;
    v33 = 1;
    v35 = 0;
    v37 = 2;
    v38 = 0;
    v39 = 1;
    sub_29D3790((__int64)&v29, (__int64)&unk_4F875F0, v17, (__int64)v34, v18, v19);
    sub_29D3790((__int64)&v29, (__int64)&unk_4F92388, v21, v22, v23, v24);
    sub_29D3790((__int64)&v29, (__int64)&unk_4F81450, v25, v26, v27, v28);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v34, (__int64)&v29);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v40, (__int64)&v35);
    if ( !v39 )
      _libc_free((unsigned __int64)v36);
    if ( !v33 )
      _libc_free((unsigned __int64)v30);
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
