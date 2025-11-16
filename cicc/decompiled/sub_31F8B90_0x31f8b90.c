// Function: sub_31F8B90
// Address: 0x31f8b90
//
__int64 __fastcall sub_31F8B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  unsigned int v7; // ebx
  int v8; // eax
  __int64 v9; // rdi
  void (*v10)(); // rax
  __int64 v11; // rdi
  void (*v12)(); // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r8
  unsigned __int8 v16; // dl
  const void *v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  char *v20; // rdi
  char *v21; // rdx
  int v22; // esi
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // esi
  int v27; // ecx
  int v28; // edx
  __int64 *v29; // rdi
  __int64 v30; // rdx
  void (*v31)(); // rcx
  __int128 *v32; // r15
  __int128 *v33; // rbx
  __int64 *v34; // rdi
  __int64 v35; // rax
  void (*v36)(); // rax
  __int64 i; // rsi
  __int64 v38; // r8
  __int64 v39; // r9
  void (*v40)(); // rax
  int v42; // [rsp+Ch] [rbp-94h]
  __int64 v43; // [rsp+10h] [rbp-90h]
  unsigned __int64 v44; // [rsp+18h] [rbp-88h]
  _DWORD v45[4]; // [rsp+20h] [rbp-80h] BYREF
  __int128 v46; // [rsp+30h] [rbp-70h] BYREF
  _OWORD v47[2]; // [rsp+40h] [rbp-60h] BYREF
  char v48; // [rsp+60h] [rbp-40h]
  char v49; // [rsp+61h] [rbp-3Fh]

  v6 = sub_31F8790(a1, 4412, a3, a4, a5);
  v7 = *(unsigned __int8 *)(a1 + 800);
  v43 = v6;
  if ( sub_BAA6A0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL), 0) )
    v7 |= 0x40000u;
  v8 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 264LL);
  if ( *(char *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 904LL) < 0 || v8 == 36 || v8 == 3 )
    BYTE1(v7) |= 0x40u;
  v9 = *(_QWORD *)(a1 + 528);
  v10 = *(void (**)())(*(_QWORD *)v9 + 120LL);
  v49 = 1;
  *(_QWORD *)&v47[0] = "Flags and language";
  v48 = 3;
  if ( v10 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _OWORD *, __int64))v10)(v9, v47, 1);
    v9 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v9 + 536LL))(v9, v7, 4);
  v11 = *(_QWORD *)(a1 + 528);
  v12 = *(void (**)())(*(_QWORD *)v11 + 120LL);
  v49 = 1;
  *(_QWORD *)&v47[0] = "CPUType";
  v48 = 3;
  if ( v12 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _OWORD *, __int64))v12)(v11, v47, 1);
    v11 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v11 + 536LL))(v11, *(unsigned __int16 *)(a1 + 786), 2);
  v13 = sub_BA8DC0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL), (__int64)"llvm.dbg.cu", 11);
  sub_B91A00(v13);
  v14 = sub_B91A10(v13, 0);
  v16 = *(_BYTE *)(v14 - 16);
  if ( (v16 & 2) == 0 )
  {
    v17 = *(const void **)(v14 - 16 - 8LL * ((v16 >> 2) & 0xF) + 8);
    if ( v17 )
      goto LABEL_13;
LABEL_39:
    v28 = 0;
    v27 = 0;
    v26 = 0;
    v25 = 0;
    v44 = 0;
    goto LABEL_23;
  }
  v17 = *(const void **)(*(_QWORD *)(v14 - 32) + 8LL);
  if ( !v17 )
    goto LABEL_39;
LABEL_13:
  v18 = sub_B91420((__int64)v17);
  v20 = (char *)(v18 + v19);
  v44 = v19;
  v17 = (const void *)v18;
  v47[0] = 0;
  if ( v18 + v19 != v18 )
  {
    v21 = (char *)v18;
    v22 = 0;
    v15 = 0xFFFF;
    while ( 1 )
    {
      v24 = *v21;
      if ( (unsigned int)(v24 - 48) <= 9 )
      {
        v23 = v24 - 48 + 10 * *((_DWORD *)v47 + v22);
        if ( v23 > 0xFFFF )
          v23 = 0xFFFF;
        *((_DWORD *)v47 + v22) = v23;
      }
      else if ( (_BYTE)v24 == 46 )
      {
        if ( ++v22 > 3 )
        {
LABEL_22:
          v25 = v47[0];
          v26 = DWORD1(v47[0]);
          v27 = DWORD2(v47[0]);
          v28 = HIDWORD(v47[0]);
          goto LABEL_23;
        }
      }
      else if ( v22 )
      {
        goto LABEL_22;
      }
      if ( v20 == ++v21 )
        goto LABEL_22;
    }
  }
  v28 = 0;
  v27 = 0;
  v26 = 0;
  v25 = 0;
LABEL_23:
  v29 = *(__int64 **)(a1 + 528);
  v45[3] = v28;
  v45[2] = v27;
  v30 = *v29;
  v45[0] = v25;
  v45[1] = v26;
  v31 = *(void (**)())(v30 + 120);
  v49 = 1;
  *(_QWORD *)&v47[0] = "Frontend version";
  v48 = 3;
  if ( v31 != nullsub_98 )
  {
    v42 = v25;
    ((void (__fastcall *)(__int64 *, _OWORD *, __int64, void (*)(), __int64))v31)(v29, v47, 1, v31, v15);
    v29 = *(__int64 **)(a1 + 528);
    v25 = v42;
  }
  v32 = (__int128 *)v45;
  v33 = &v46;
  while ( 1 )
  {
    v32 = (__int128 *)((char *)v32 + 4);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v29 + 536))(v29, v25, 2);
    if ( v32 == &v46 )
      break;
    v29 = *(__int64 **)(a1 + 528);
    v25 = *(_DWORD *)v32;
  }
  v34 = *(__int64 **)(a1 + 528);
  v46 = 0;
  v35 = *v34;
  LODWORD(v46) = 20000;
  v36 = *(void (**)())(v35 + 120);
  v49 = 1;
  *(_QWORD *)&v47[0] = "Backend version";
  v48 = 3;
  if ( v36 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, _OWORD *, __int64))v36)(v34, v47, 1);
    v34 = *(__int64 **)(a1 + 528);
  }
  for ( i = 20000; ; i = *(int *)v33 )
  {
    v33 = (__int128 *)((char *)v33 + 4);
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v34 + 536))(v34, i, 2);
    v34 = *(__int64 **)(a1 + 528);
    if ( v33 == v47 )
      break;
  }
  v40 = *(void (**)())(*v34 + 120);
  v49 = 1;
  *(_QWORD *)&v47[0] = "Null-terminated compiler version string";
  v48 = 3;
  if ( v40 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, __int128 *, __int64))v40)(v34, v33, 1);
    v34 = *(__int64 **)(a1 + 528);
  }
  sub_31F4F00(v34, v17, v44, 3840, v38, v39);
  return sub_31F8930(a1, v43);
}
