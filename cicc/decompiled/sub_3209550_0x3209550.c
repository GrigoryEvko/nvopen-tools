// Function: sub_3209550
// Address: 0x3209550
//
__int64 __fastcall sub_3209550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdi
  void (*v11)(); // rcx
  unsigned __int8 v12; // cl
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // rdi
  void (*v19)(); // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int8 v23; // dl
  const void *v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 result; // rax
  __int64 v28; // r15
  __int64 v29; // r12
  int v30; // ecx
  unsigned __int16 v31; // r14
  __int64 v32; // r8
  unsigned int v33; // r13d
  int v34; // ecx
  unsigned __int16 v35; // r14
  char v36; // al
  __int16 v37; // dx
  __int64 *v38; // r9
  __int64 v39; // rsi
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v43; // [rsp+10h] [rbp-80h]
  __int16 v44; // [rsp+1Ah] [rbp-76h]
  int v45; // [rsp+1Ch] [rbp-74h]
  unsigned int v46; // [rsp+24h] [rbp-6Ch]
  __int64 v47; // [rsp+28h] [rbp-68h]
  __int64 v48; // [rsp+28h] [rbp-68h]
  char *v49; // [rsp+30h] [rbp-60h] BYREF
  __int64 v50; // [rsp+38h] [rbp-58h]
  __int64 v51; // [rsp+40h] [rbp-50h]
  char v52; // [rsp+48h] [rbp-48h] BYREF
  char v53; // [rsp+50h] [rbp-40h]
  char v54; // [rsp+51h] [rbp-3Fh]

  v7 = sub_31F8790(a1, 4414, a3, a4, a5);
  v8 = *(_QWORD *)a3;
  v9 = *(_WORD *)(*(_QWORD *)a3 + 20LL) != 0;
  if ( !*(_DWORD *)(a3 + 48) )
    LOWORD(v9) = v9 | 0x100;
  v10 = *(_QWORD *)(a1 + 528);
  v11 = *(void (**)())(*(_QWORD *)v10 + 120LL);
  v54 = 1;
  v49 = "TypeIndex";
  v53 = 3;
  if ( v11 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v11)(v10, &v49, 1);
    v8 = *(_QWORD *)a3;
  }
  v12 = *(_BYTE *)(v8 - 16);
  v13 = v8 - 16;
  if ( *(_BYTE *)(a3 + 56) )
  {
    if ( (v12 & 2) != 0 )
      v14 = *(_QWORD *)(v8 - 32);
    else
      v14 = v13 - 8LL * ((v12 >> 2) & 0xF);
    v46 = sub_3208510(a1, *(_QWORD *)(v14 + 24));
  }
  else
  {
    if ( (v12 & 2) != 0 )
      v15 = *(_QWORD *)(v8 - 32);
    else
      v15 = v13 - 8LL * ((v12 >> 2) & 0xF);
    v16 = *(_QWORD *)(v15 + 24);
    if ( v16 )
    {
      v17 = sub_3205010(a1, v16);
    }
    else
    {
      LODWORD(v49) = 3;
      v17 = 3;
    }
    v46 = v17;
  }
  (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 528) + 536LL))(*(_QWORD *)(a1 + 528), v46, 4);
  v18 = *(_QWORD *)(a1 + 528);
  v19 = *(void (**)())(*(_QWORD *)v18 + 120LL);
  v54 = 1;
  v49 = "Flags";
  v53 = 3;
  if ( v19 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v19)(v18, &v49, 1);
    v18 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v18 + 536LL))(v18, (unsigned __int16)v9, 2);
  v22 = *(_QWORD *)a3;
  v23 = *(_BYTE *)(*(_QWORD *)a3 - 16LL);
  if ( (v23 & 2) != 0 )
  {
    v24 = *(const void **)(*(_QWORD *)(v22 - 32) + 8LL);
    if ( v24 )
    {
LABEL_18:
      v24 = (const void *)sub_B91420((__int64)v24);
      goto LABEL_19;
    }
  }
  else
  {
    v24 = *(const void **)(v22 - 16 - 8LL * ((v23 >> 2) & 0xF) + 8);
    if ( v24 )
      goto LABEL_18;
  }
  v25 = 0;
LABEL_19:
  sub_31F4F00(*(__int64 **)(a1 + 528), v24, v25, 3840, v20, v21);
  sub_31F8930(a1, v7);
  v26 = *(unsigned int *)(a3 + 48);
  v50 = 0;
  v49 = &v52;
  result = *(_QWORD *)(a3 + 40);
  v51 = 20;
  v28 = result + 40 * v26;
  v44 = v9 & 1;
  v29 = result;
  if ( v28 != result )
  {
    while ( 1 )
    {
      v30 = *(_DWORD *)v29;
      v31 = *(_WORD *)(v29 + 4);
      v32 = *(_QWORD *)v29;
      v33 = *(unsigned __int16 *)(v29 + 6);
      v50 = 0;
      v34 = v30 >> 1;
      v35 = v31 >> 1;
      if ( (v32 & 1) == 0 )
      {
        v38 = *(__int64 **)(a1 + 528);
        v39 = *(_QWORD *)(v29 + 8);
        v40 = *(unsigned int *)(v29 + 16);
        v41 = *v38;
        LODWORD(v48) = (unsigned __int16)v33;
        if ( (v32 & 0x100000000LL) != 0 )
        {
          HIDWORD(v48) = v35;
          result = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64))(v41 + 776))(v38, v39, v40, v48);
        }
        else
        {
          result = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD))(v41 + 784))(
                     v38,
                     v39,
                     v40,
                     (unsigned __int16)v33);
        }
        goto LABEL_29;
      }
      if ( (_WORD)v33 == 21 )
      {
        v33 = 30006;
        v34 += *(_DWORD *)(a2 + 476);
      }
      v43 = v32;
      v45 = v34;
      v36 = sub_370BBB0(v33, *(unsigned __int16 *)(a1 + 786));
      if ( (v43 & 0x100000000LL) != 0 )
      {
        v37 = (16 * v35) | 1;
        goto LABEL_28;
      }
      if ( !v36 )
        goto LABEL_27;
      v37 = v44;
      if ( v44 )
        break;
      if ( *(_BYTE *)(a2 + 480) == v36 )
      {
LABEL_34:
        result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 528) + 792LL))(
                   *(_QWORD *)(a1 + 528),
                   *(_QWORD *)(v29 + 8),
                   *(unsigned int *)(v29 + 16));
        goto LABEL_29;
      }
LABEL_28:
      WORD1(v47) = v37;
      LOWORD(v47) = v33;
      HIDWORD(v47) = v45;
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 528) + 768LL))(
                 *(_QWORD *)(a1 + 528),
                 *(_QWORD *)(v29 + 8),
                 *(unsigned int *)(v29 + 16),
                 v47);
LABEL_29:
      v29 += 40;
      if ( v28 == v29 )
        return result;
    }
    if ( *(_BYTE *)(a2 + 481) == v36 )
      goto LABEL_34;
LABEL_27:
    v37 = 0;
    goto LABEL_28;
  }
  return result;
}
