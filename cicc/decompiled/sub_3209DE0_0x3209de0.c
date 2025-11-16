// Function: sub_3209DE0
// Address: 0x3209de0
//
__int64 __fastcall sub_3209DE0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v8; // rdx
  unsigned int v9; // esi
  __int64 v10; // rdi
  int v11; // r11d
  __int64 *v12; // rcx
  unsigned int i; // eax
  _QWORD *v14; // r8
  __int64 v15; // r10
  unsigned int v16; // eax
  _DWORD *v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r14
  void (*v21)(); // rax
  __int64 v22; // rdi
  void (*v23)(); // rax
  __int64 v24; // rdi
  void (*v25)(); // rax
  _BYTE *v26; // rsi
  unsigned __int8 v27; // al
  _BYTE **v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // r8
  __int64 v31; // r9
  _QWORD *v32; // r14
  _QWORD *j; // r13
  __int64 v34; // r8
  unsigned __int64 v35; // r9
  _QWORD *v36; // rcx
  _QWORD *v37; // rax
  _QWORD *v38; // rdi
  int v40; // eax
  int v41; // edi
  __int64 v42; // rax
  __int64 *v43; // [rsp+8h] [rbp-68h] BYREF
  char *v44; // [rsp+10h] [rbp-60h] BYREF
  __int64 v45; // [rsp+18h] [rbp-58h]
  char v46; // [rsp+30h] [rbp-40h]
  char v47; // [rsp+31h] [rbp-3Fh]

  v4 = a1 + 1216;
  v8 = *(_QWORD *)(a4 + 128);
  v9 = *(_DWORD *)(a1 + 1240);
  v45 = 0;
  v44 = (char *)v8;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 1216);
    v43 = 0;
    goto LABEL_44;
  }
  v10 = *(_QWORD *)(a1 + 1224);
  v11 = 1;
  v12 = 0;
  for ( i = (v9 - 1) & (969526130 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))); ; i = (v9 - 1) & v16 )
  {
    v14 = (_QWORD *)(v10 + 24LL * i);
    v15 = *v14;
    if ( v8 == *v14 && !v14[1] )
    {
      v17 = v14 + 2;
      goto LABEL_12;
    }
    if ( v15 == -4096 )
      break;
    if ( v15 == -8192 && v14[1] == -8192 && !v12 )
      v12 = (__int64 *)(v10 + 24LL * i);
LABEL_9:
    v16 = v11 + i;
    ++v11;
  }
  if ( v14[1] != -4096 )
    goto LABEL_9;
  v40 = *(_DWORD *)(a1 + 1232);
  if ( !v12 )
    v12 = v14;
  ++*(_QWORD *)(a1 + 1216);
  v41 = v40 + 1;
  v43 = v12;
  if ( 4 * (v40 + 1) < 3 * v9 )
  {
    if ( v9 - *(_DWORD *)(a1 + 1236) - v41 > v9 >> 3 )
      goto LABEL_38;
    goto LABEL_45;
  }
LABEL_44:
  v9 *= 2;
LABEL_45:
  sub_31FE9B0(v4, v9);
  sub_31FB320(v4, (__int64 *)&v44, &v43);
  v8 = (__int64)v44;
  v12 = v43;
  v41 = *(_DWORD *)(a1 + 1232) + 1;
LABEL_38:
  *(_DWORD *)(a1 + 1232) = v41;
  if ( *v12 != -4096 || v12[1] != -4096 )
    --*(_DWORD *)(a1 + 1236);
  *v12 = v8;
  v42 = v45;
  v17 = v12 + 2;
  *((_DWORD *)v12 + 4) = 0;
  v12[1] = v42;
LABEL_12:
  LODWORD(v43) = *v17;
  v18 = sub_31F8790(a1, 4429, v8, (__int64)v12, (__int64)v17);
  v19 = *(_QWORD *)(a1 + 528);
  v20 = v18;
  v21 = *(void (**)())(*(_QWORD *)v19 + 120LL);
  v47 = 1;
  v44 = "PtrParent";
  v46 = 3;
  if ( v21 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v21)(v19, &v44, 1);
    v19 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v19 + 536LL))(v19, 0, 4);
  v22 = *(_QWORD *)(a1 + 528);
  v23 = *(void (**)())(*(_QWORD *)v22 + 120LL);
  v47 = 1;
  v44 = "PtrEnd";
  v46 = 3;
  if ( v23 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v23)(v22, &v44, 1);
    v22 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v22 + 536LL))(v22, 0, 4);
  v24 = *(_QWORD *)(a1 + 528);
  v25 = *(void (**)())(*(_QWORD *)v24 + 120LL);
  v47 = 1;
  v44 = "Inlinee type index";
  v46 = 3;
  if ( v25 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v25)(v24, &v44, 1);
    v24 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v24 + 536LL))(v24, (unsigned int)v43, 4);
  v26 = *(_BYTE **)(a4 + 128);
  if ( *v26 != 16 )
  {
    v27 = *(v26 - 16);
    if ( (v27 & 2) != 0 )
      v28 = (_BYTE **)*((_QWORD *)v26 - 4);
    else
      v28 = (_BYTE **)&v26[-8 * ((v27 >> 2) & 0xF) - 16];
    v26 = *v28;
  }
  v29 = sub_31FF830(a1, (unsigned __int64)v26);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 528) + 752LL))(
    *(_QWORD *)(a1 + 528),
    *(unsigned int *)(a4 + 136),
    v29,
    *(unsigned int *)(*(_QWORD *)(a4 + 128) + 16LL),
    a2[55],
    a2[56]);
  sub_31F8930(a1, v20);
  sub_3209920(a1, (__int64)a2, *(_QWORD *)a4, *(unsigned int *)(a4 + 8), v30, v31);
  v32 = *(_QWORD **)(a4 + 104);
  for ( j = &v32[*(unsigned int *)(a4 + 112)]; j != v32; ++v32 )
  {
    v34 = *v32;
    v35 = a2[1];
    v36 = *(_QWORD **)(*a2 + 8 * (*v32 % v35));
    if ( v36 )
    {
      v37 = (_QWORD *)*v36;
      if ( v34 == *(_QWORD *)(*v36 + 8LL) )
      {
LABEL_28:
        v36 = (_QWORD *)*v36;
      }
      else
      {
        while ( 1 )
        {
          v38 = (_QWORD *)*v37;
          if ( !*v37 )
            break;
          v36 = v37;
          if ( *v32 % v35 != v38[1] % v35 )
            break;
          v37 = (_QWORD *)*v37;
          if ( v34 == v38[1] )
            goto LABEL_28;
        }
        v36 = 0;
      }
    }
    sub_3209DE0(a1, a2, *v32, v36 + 2);
  }
  return sub_31F93A0(a1, 0x114Eu);
}
