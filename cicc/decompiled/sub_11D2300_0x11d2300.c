// Function: sub_11D2300
// Address: 0x11d2300
//
__int64 __fastcall sub_11D2300(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  void *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r13
  int v14; // r10d
  unsigned int i; // eax
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // r12
  __int64 *v19; // rax
  __int64 *v20; // r14
  int v21; // ebx
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rsi
  __int64 *v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v38; // [rsp+18h] [rbp-88h]
  int v39; // [rsp+20h] [rbp-80h]
  int v40; // [rsp+24h] [rbp-7Ch]
  int v41; // [rsp+28h] [rbp-78h]
  char v42; // [rsp+2Ch] [rbp-74h]
  _QWORD v43[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v44; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v45; // [rsp+48h] [rbp-58h]
  __int64 v46; // [rsp+50h] [rbp-50h]
  int v47; // [rsp+58h] [rbp-48h]
  char v48; // [rsp+5Ch] [rbp-44h]
  _BYTE v49[64]; // [rsp+60h] [rbp-40h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v7 = v6 + 8;
  v8 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v11 = *(unsigned int *)(a4 + 88);
  v12 = *(_QWORD *)(a4 + 72);
  v13 = v8 + 8;
  if ( !(_DWORD)v11 )
    goto LABEL_20;
  v14 = 1;
  v9 = (unsigned int)(v11 - 1);
  for ( i = v9
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v9 & v17 )
  {
    v16 = v12 + 24LL * i;
    v10 = *(void **)v16;
    if ( *(_UNKNOWN **)v16 == &unk_4F881D0 && a3 == *(_QWORD *)(v16 + 8) )
      break;
    if ( v10 == (void *)-4096LL && *(_QWORD *)(v16 + 8) == -4096 )
      goto LABEL_20;
    v17 = v14 + i;
    ++v14;
  }
  if ( v16 == v12 + 24 * v11 )
  {
LABEL_20:
    v18 = 0;
  }
  else
  {
    v18 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL);
    if ( v18 )
      v18 += 8;
  }
  v19 = *(__int64 **)(v6 + 48);
  v20 = *(__int64 **)(v6 + 40);
  v21 = 0;
  v36 = v19;
  if ( v20 == v19 )
    goto LABEL_16;
  do
  {
    v22 = *v20++;
    v21 |= sub_11D2180(v22, v13, v7, v18, v9, (__int64)v10);
  }
  while ( v36 != v20 );
  if ( !(_BYTE)v21 )
  {
LABEL_16:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &unk_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v42 = 1;
  v38 = v43;
  v43[0] = &unk_4F82408;
  v39 = 2;
  v41 = 0;
  v44 = 0;
  v45 = v49;
  v46 = 2;
  v47 = 0;
  v48 = 1;
  v40 = 1;
  v37 = 1;
  sub_11CDDF0((__int64)&v37, (__int64)&unk_4F881D0, v23, v24, v9, (__int64)v10);
  sub_11CDDF0((__int64)&v37, (__int64)&unk_4F8E5A8, v25, v26, v27, v28);
  sub_11CDDF0((__int64)&v37, (__int64)&unk_4F8F810, v29, v30, v31, v32);
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v43, (__int64)&v37);
  v33 = a1 + 80;
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v49, (__int64)&v44);
  if ( !v48 )
  {
    _libc_free(v45, v33);
    if ( v42 )
      return a1;
LABEL_18:
    _libc_free(v38, v33);
    return a1;
  }
  if ( !v42 )
    goto LABEL_18;
  return a1;
}
