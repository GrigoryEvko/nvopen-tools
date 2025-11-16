// Function: sub_31F1320
// Address: 0x31f1320
//
__int64 *__fastcall sub_31F1320(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 *v5; // rax
  __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  unsigned __int16 v8; // r15
  const char *v9; // rax
  __int64 v10; // rdx
  void (*v11)(); // r15
  const char *v12; // rax
  __int64 v13; // rdx
  __int64 *result; // rax
  __int64 v15; // rdi
  void (*v16)(); // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 *v19; // rbx
  __int64 v20; // r15
  __int64 v21; // rdi
  void (*v22)(); // rax
  __int64 v23; // rax
  __int64 v24; // r13
  void (*v25)(); // rbx
  const char *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-1A0h]
  void (*v31)(); // [rsp+8h] [rbp-198h]
  __int64 v32; // [rsp+8h] [rbp-198h]
  void (__fastcall *v33)(__int64, _QWORD, __int64, __int64, _QWORD); // [rsp+8h] [rbp-198h]
  __int64 v34; // [rsp+10h] [rbp-190h] BYREF
  __int64 v35; // [rsp+18h] [rbp-188h] BYREF
  const char *v36; // [rsp+20h] [rbp-180h] BYREF
  int v37; // [rsp+30h] [rbp-170h]
  __int16 v38; // [rsp+40h] [rbp-160h]
  _QWORD v39[4]; // [rsp+50h] [rbp-150h] BYREF
  __int16 v40; // [rsp+70h] [rbp-130h]
  _QWORD v41[4]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v42; // [rsp+A0h] [rbp-100h]
  _QWORD v43[4]; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v44; // [rsp+D0h] [rbp-D0h]
  _QWORD v45[4]; // [rsp+E0h] [rbp-C0h] BYREF
  __int16 v46; // [rsp+100h] [rbp-A0h]
  _QWORD v47[4]; // [rsp+110h] [rbp-90h] BYREF
  __int16 v48; // [rsp+130h] [rbp-70h]
  const char *v49; // [rsp+140h] [rbp-60h] BYREF
  __int64 v50; // [rsp+148h] [rbp-58h]
  const char *v51; // [rsp+150h] [rbp-50h]
  __int64 v52; // [rsp+158h] [rbp-48h]
  __int16 v53; // [rsp+160h] [rbp-40h]

  if ( !*(_BYTE *)(a1 + 488) )
    goto LABEL_2;
  v24 = *(_QWORD *)(a1 + 224);
  v25 = *(void (**)())(*(_QWORD *)v24 + 120LL);
  v26 = sub_E02B90(*(unsigned __int16 *)(a2 + 28));
  v27 = *(unsigned int *)(a2 + 20);
  v4 = *(unsigned int *)(a2 + 24);
  v51 = v26;
  v35 = v27;
  v28 = *(unsigned int *)(a2 + 16);
  v37 = v4;
  v34 = v28;
  v36 = "Abbrev [";
  v39[0] = &v36;
  v39[2] = "] 0x";
  v41[0] = v39;
  v41[2] = &v34;
  v43[0] = v41;
  v43[2] = ":0x";
  v45[0] = v43;
  v45[2] = &v35;
  v47[0] = v45;
  v47[2] = " ";
  v38 = 2307;
  v40 = 770;
  v42 = 3842;
  v44 = 770;
  v46 = 3842;
  v48 = 770;
  v49 = (const char *)v47;
  v52 = v29;
  v53 = 1282;
  if ( v25 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v25)(v24, &v49, 1);
LABEL_2:
    v4 = *(unsigned int *)(a2 + 24);
  }
  (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 424LL))(a1, v4, 0, 0);
  v5 = *(__int64 **)(a2 + 8);
  if ( v5 )
  {
    v6 = *v5;
    do
    {
      v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v7 )
        break;
      v8 = *(_WORD *)(v7 + 12);
      if ( !*(_BYTE *)(a1 + 488) )
        goto LABEL_26;
      v30 = *(_QWORD *)(a1 + 224);
      v31 = *(void (**)())(*(_QWORD *)v30 + 120LL);
      v9 = sub_E058A0(v8);
      v53 = 261;
      v49 = v9;
      v50 = v10;
      if ( v31 != nullsub_98 )
        ((void (__fastcall *)(__int64, const char **, __int64))v31)(v30, &v49, 1);
      if ( v8 == 50 )
      {
        v32 = *(_QWORD *)(a1 + 224);
        v11 = *(void (**)())(*(_QWORD *)v32 + 120LL);
        v12 = sub_E0A520(*(_DWORD *)(v7 + 16));
        v53 = 261;
        v49 = v12;
        v50 = v13;
        if ( v11 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v11)(v32, &v49, 1);
      }
      else
      {
LABEL_26:
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 200) + 544LL) - 42) <= 1 && v8 == 2 && *(_DWORD *)(v7 + 8) == 1 )
        {
          v20 = *(_QWORD *)(v7 + 16);
          v21 = *(_QWORD *)(a1 + 224);
          v47[0] = v20;
          v22 = *(void (**)())(*(_QWORD *)v21 + 120LL);
          v49 = "debug_loc offset ";
          v51 = (const char *)v47;
          v53 = 2819;
          if ( v22 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v22)(v21, &v49, 1);
            v20 = v47[0];
          }
          v33 = *(void (__fastcall **)(__int64, _QWORD, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 432LL);
          v23 = sub_31DA6B0(a1);
          v33(a1, *(_QWORD *)(*(_QWORD *)(v23 + 144) + 16LL), v20, 4, 0);
          goto LABEL_8;
        }
      }
      sub_3215FD0(v7 + 8, a1);
LABEL_8:
      v6 = *(_QWORD *)v7;
    }
    while ( (v6 & 4) == 0 );
  }
  result = *(__int64 **)(a2 + 32);
  if ( !*(_BYTE *)(a2 + 30) )
  {
    if ( !result )
      return result;
    goto LABEL_22;
  }
  if ( result )
  {
LABEL_22:
    v17 = *result;
    do
    {
      v18 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      v19 = (__int64 *)v18;
      if ( !v18 )
        break;
      sub_31F1320(a1, v18);
      v17 = *v19;
    }
    while ( (*v19 & 4) == 0 );
  }
  v15 = *(_QWORD *)(a1 + 224);
  v16 = *(void (**)())(*(_QWORD *)v15 + 120LL);
  v49 = "End Of Children Mark";
  v53 = 259;
  if ( v16 != nullsub_98 )
    ((void (__fastcall *)(__int64, const char **, __int64))v16)(v15, &v49, 1);
  return (__int64 *)sub_31DC9D0(a1, 0);
}
