// Function: sub_2DB1790
// Address: 0x2db1790
//
__int64 __fastcall sub_2DB1790(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 (*v5)(); // rax
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // r10d
  unsigned int i; // eax
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rsi
  int v20; // r10d
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 *v23; // rcx
  __int64 v24; // rcx
  __int64 **v25; // r8
  __int64 v26; // rax
  void *v27; // rsi
  int v29; // [rsp+4h] [rbp-9Ch]
  int v30; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v31; // [rsp+8h] [rbp-98h]
  __int64 v32; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v33; // [rsp+18h] [rbp-88h]
  int v34; // [rsp+20h] [rbp-80h]
  int v35; // [rsp+24h] [rbp-7Ch]
  int v36; // [rsp+28h] [rbp-78h]
  char v37; // [rsp+2Ch] [rbp-74h]
  _QWORD v38[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v39; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v40; // [rsp+48h] [rbp-58h]
  __int64 v41; // [rsp+50h] [rbp-50h]
  int v42; // [rsp+58h] [rbp-48h]
  char v43; // [rsp+5Ch] [rbp-44h]
  _BYTE v44[64]; // [rsp+60h] [rbp-40h] BYREF

  v5 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
  if ( v5 == sub_23CE270 )
    BUG();
  v9 = 0;
  v10 = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(*a2, a3);
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 144LL);
  if ( v11 != sub_2C8F680 )
  {
    v18 = ((__int64 (__fastcall *)(__int64))v11)(v10);
    v12 = *(unsigned int *)(a4 + 88);
    v13 = *(_QWORD *)(a4 + 72);
    v9 = v18;
    if ( (_DWORD)v12 )
      goto LABEL_4;
LABEL_9:
    v19 = *a2;
    goto LABEL_10;
  }
  v12 = *(unsigned int *)(a4 + 88);
  v13 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v12 )
    goto LABEL_9;
LABEL_4:
  v14 = 1;
  for ( i = (v12 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v12 - 1) & v17 )
  {
    v16 = v13 + 24LL * i;
    if ( *(_UNKNOWN **)v16 == &unk_4F81450 && a3 == *(_QWORD *)(v16 + 8) )
      break;
    if ( *(_QWORD *)v16 == -4096 && *(_QWORD *)(v16 + 8) == -4096 )
      goto LABEL_9;
    v17 = v14 + i;
    ++v14;
  }
  v21 = *a2;
  v20 = *(_DWORD *)(*a2 + 648LL);
  v19 = *a2;
  if ( v16 != v13 + 24 * v12 )
  {
    v24 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL);
    if ( v24 )
    {
      v23 = (unsigned __int64 *)(v24 + 8);
      v25 = 0;
      if ( !v20 )
        goto LABEL_17;
      goto LABEL_16;
    }
  }
LABEL_10:
  v20 = *(_DWORD *)(v19 + 648);
  v21 = v19;
  if ( v20 )
  {
    v30 = *(_DWORD *)(v19 + 648);
    v22 = sub_BC1CD0(a4, &unk_4F81450, a3);
    v20 = v30;
    v23 = (unsigned __int64 *)(v22 + 8);
LABEL_16:
    v29 = v20;
    v31 = v23;
    v26 = sub_BC1CD0(a4, &unk_4F89C30, a3);
    v20 = v29;
    v23 = v31;
    v25 = (__int64 **)(v26 + 8);
    v21 = *a2;
    goto LABEL_17;
  }
  v25 = 0;
  v23 = 0;
LABEL_17:
  v27 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2DB0210(v20, a3, v9, v23, v25, 0, (_DWORD *)(v21 + 512)) )
  {
    v33 = v38;
    v34 = 2;
    v38[0] = &unk_4F81450;
    v36 = 0;
    v37 = 1;
    v39 = 0;
    v40 = v44;
    v41 = 2;
    v42 = 0;
    v43 = 1;
    v35 = 1;
    v32 = 1;
    sub_C8CF70(a1, v27, 2, (__int64)v38, (__int64)&v32);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v44, (__int64)&v39);
    if ( !v43 )
      _libc_free((unsigned __int64)v40);
    if ( !v37 )
      _libc_free((unsigned __int64)v33);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v27;
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
