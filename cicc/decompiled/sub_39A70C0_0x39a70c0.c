// Function: sub_39A70C0
// Address: 0x39a70c0
//
void __fastcall sub_39A70C0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  char v8; // r15
  __int64 v9; // rdi
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r14
  __int64 v15; // rax
  char v16; // al
  char v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 (*v20)(void); // rax
  int v21; // eax
  int v22; // edx
  void *v23; // rcx
  size_t v24; // rdx
  size_t v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 *v28; // r15
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // r9
  unsigned int v33; // r8d
  __int64 *v34; // rax
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 *v37; // rcx
  int v38; // eax
  int v39; // edi
  int v40; // eax
  int v41; // r8d
  __int64 v42; // rsi
  unsigned int v43; // eax
  __int64 v44; // r9
  int v45; // r11d
  __int64 *v46; // r10
  int v47; // eax
  int v48; // eax
  __int64 *v49; // r9
  int v50; // r10d
  unsigned int v51; // r15d
  __int64 v52; // r8
  __int64 v53; // rsi
  int v54; // [rsp+8h] [rbp-48h]
  __int64 v55; // [rsp+8h] [rbp-48h]
  __int64 v56; // [rsp+8h] [rbp-48h]
  _DWORD v57[13]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( !a4 || *(_BYTE *)(*(_QWORD *)(a1 + 80) + 49LL) )
  {
    v8 = sub_39A6E10((__int64 *)a1, a2, a3);
    if ( v8 )
      return;
  }
  else
  {
    v8 = a4;
  }
  v9 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
  if ( !v9 || (sub_161E970(v9), !v10) )
  {
    if ( v8 )
      goto LABEL_7;
    goto LABEL_44;
  }
  v23 = *(void **)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
  if ( v23 )
  {
    v23 = (void *)sub_161E970((__int64)v23);
    v25 = v24;
  }
  else
  {
    v25 = 0;
  }
  sub_39A3F30((__int64 *)a1, a3, 3, v23, v25);
  if ( !v8 )
LABEL_44:
    sub_39A3770(a1, a3, a2);
LABEL_7:
  if ( a4 )
    return;
  if ( (*(_BYTE *)(a2 + 45) & 1) != 0 )
  {
    v11 = *(_DWORD *)(*(_QWORD *)(a1 + 80) + 24LL);
    if ( (((_WORD)v11 - 12) & 0xFFFB) == 0 || (_WORD)v11 == 1 )
      sub_39A34D0(a1, a3, 39);
  }
  v12 = *(_QWORD *)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8)));
  if ( !v12 )
  {
    v14 = 0;
    goto LABEL_17;
  }
  v13 = *(unsigned __int8 *)(v12 + 52);
  v14 = *(_QWORD *)(v12 + 8 * (3LL - *(unsigned int *)(v12 + 8)));
  if ( !v14 )
  {
    if ( (unsigned __int8)v13 > 1u )
    {
      v57[0] = 65547;
      sub_39A3560(a1, (__int64 *)(a3 + 8), 54, (__int64)v57, v13);
    }
LABEL_17:
    v16 = *(_BYTE *)(a2 + 40);
    v17 = v16 & 3;
    if ( (v16 & 3) != 0 )
      goto LABEL_47;
    goto LABEL_18;
  }
  if ( (unsigned __int8)v13 > 1u )
  {
    v57[0] = 65547;
    sub_39A3560(a1, (__int64 *)(a3 + 8), 54, (__int64)v57, v13);
  }
  v15 = *(unsigned int *)(v14 + 8);
  if ( !(_DWORD)v15 )
    goto LABEL_17;
  v26 = *(_QWORD *)(v14 - 8 * v15);
  if ( !v26 )
    goto LABEL_17;
  sub_39A6760((__int64 *)a1, a3, v26, 73);
  v16 = *(_BYTE *)(a2 + 40);
  v17 = v16 & 3;
  if ( (v16 & 3) != 0 )
  {
LABEL_47:
    v57[0] = 65547;
    sub_39A3560(a1, (__int64 *)(a3 + 8), 76, (__int64)v57, v17 & 3);
    if ( *(_DWORD *)(a2 + 32) != -1 )
    {
      v27 = sub_145CBF0((__int64 *)(a1 + 88), 16, 16);
      *(_QWORD *)v27 = 0;
      v28 = (__int64 *)v27;
      *(_DWORD *)(v27 + 8) = 0;
      sub_39A35E0(a1, (__int64 *)v27, 11, 16);
      sub_39A35E0(a1, v28, 15, *(unsigned int *)(a2 + 32));
      sub_39A4520((__int64 *)a1, a3, 77, (__int64 **)v28);
    }
    v29 = *(unsigned int *)(a2 + 8);
    v30 = 0;
    if ( (unsigned int)v29 > 8 )
      v30 = *(_QWORD *)(a2 + 8 * (8 - v29));
    v31 = *(_DWORD *)(a1 + 328);
    if ( v31 )
    {
      v32 = *(_QWORD *)(a1 + 312);
      v33 = (v31 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v34 = (__int64 *)(v32 + 16LL * v33);
      v35 = *v34;
      if ( a3 == *v34 )
        goto LABEL_53;
      v54 = 1;
      v37 = 0;
      while ( v35 != -8 )
      {
        if ( !v37 && v35 == -16 )
          v37 = v34;
        v33 = (v31 - 1) & (v54 + v33);
        v34 = (__int64 *)(v32 + 16LL * v33);
        v35 = *v34;
        if ( a3 == *v34 )
          goto LABEL_53;
        ++v54;
      }
      if ( !v37 )
        v37 = v34;
      v38 = *(_DWORD *)(a1 + 320);
      ++*(_QWORD *)(a1 + 304);
      v39 = v38 + 1;
      if ( 4 * (v38 + 1) < 3 * v31 )
      {
        if ( v31 - *(_DWORD *)(a1 + 324) - v39 > v31 >> 3 )
        {
LABEL_74:
          *(_DWORD *)(a1 + 320) = v39;
          if ( *v37 != -8 )
            --*(_DWORD *)(a1 + 324);
          *v37 = a3;
          v37[1] = v30;
LABEL_53:
          if ( (*(_BYTE *)(a2 + 40) & 8) != 0 )
            goto LABEL_19;
          goto LABEL_54;
        }
        v56 = v30;
        sub_39A5FB0(a1 + 304, v31);
        v47 = *(_DWORD *)(a1 + 328);
        if ( v47 )
        {
          v48 = v47 - 1;
          v49 = 0;
          v30 = v56;
          v50 = 1;
          v51 = v48 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v52 = *(_QWORD *)(a1 + 312);
          v39 = *(_DWORD *)(a1 + 320) + 1;
          v37 = (__int64 *)(v52 + 16LL * v51);
          v53 = *v37;
          if ( a3 != *v37 )
          {
            while ( v53 != -8 )
            {
              if ( !v49 && v53 == -16 )
                v49 = v37;
              v51 = v48 & (v50 + v51);
              v37 = (__int64 *)(v52 + 16LL * v51);
              v53 = *v37;
              if ( a3 == *v37 )
                goto LABEL_74;
              ++v50;
            }
            if ( v49 )
              v37 = v49;
          }
          goto LABEL_74;
        }
LABEL_106:
        ++*(_DWORD *)(a1 + 320);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 304);
    }
    v55 = v30;
    sub_39A5FB0(a1 + 304, 2 * v31);
    v40 = *(_DWORD *)(a1 + 328);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 312);
      v30 = v55;
      v43 = (v40 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v39 = *(_DWORD *)(a1 + 320) + 1;
      v37 = (__int64 *)(v42 + 16LL * v43);
      v44 = *v37;
      if ( a3 != *v37 )
      {
        v45 = 1;
        v46 = 0;
        while ( v44 != -8 )
        {
          if ( !v46 && v44 == -16 )
            v46 = v37;
          v43 = v41 & (v45 + v43);
          v37 = (__int64 *)(v42 + 16LL * v43);
          v44 = *v37;
          if ( a3 == *v37 )
            goto LABEL_74;
          ++v45;
        }
        if ( v46 )
          v37 = v46;
      }
      goto LABEL_74;
    }
    goto LABEL_106;
  }
LABEL_18:
  if ( (v16 & 8) != 0 )
    goto LABEL_19;
LABEL_54:
  sub_39A34D0(a1, a3, 60);
  sub_39A6830((__int64 *)a1, a3, v14);
LABEL_19:
  v18 = *(unsigned int *)(a2 + 8);
  v19 = 0;
  if ( (unsigned int)v18 > 0xA )
    v19 = *(_QWORD *)(a2 + 8 * (10 - v18));
  sub_39A67A0((__int64 *)a1, a3, v19);
  if ( (*(_BYTE *)(a2 + 44) & 0x40) != 0 )
    sub_39A34D0(a1, a3, 52);
  if ( (*(_BYTE *)(a2 + 40) & 4) == 0 )
    sub_39A34D0(a1, a3, 63);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 200) + 4512LL) )
  {
    if ( (*(_BYTE *)(a2 + 40) & 0x10) != 0 )
      sub_39A34D0(a1, a3, 16353);
    v20 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 192) + 336LL);
    if ( v20 != sub_214ABD0 )
    {
      v36 = v20();
      if ( v36 )
      {
        v57[0] = 65548;
        sub_39A3560(a1, (__int64 *)(a3 + 8), 16355, (__int64)v57, v36);
      }
    }
  }
  v21 = *(_DWORD *)(a2 + 44);
  if ( (v21 & 0x2000) != 0 )
  {
    sub_39A34D0(a1, a3, 119);
    v21 = *(_DWORD *)(a2 + 44);
    if ( (v21 & 0x4000) == 0 )
    {
LABEL_31:
      if ( (v21 & 0x100000) == 0 )
        goto LABEL_32;
      goto LABEL_58;
    }
  }
  else if ( (v21 & 0x4000) == 0 )
  {
    goto LABEL_31;
  }
  sub_39A34D0(a1, a3, 120);
  v21 = *(_DWORD *)(a2 + 44);
  if ( (v21 & 0x100000) == 0 )
  {
LABEL_32:
    v22 = v21 & 3;
    if ( v22 != 2 )
      goto LABEL_33;
LABEL_59:
    v57[0] = 65547;
    sub_39A3560(a1, (__int64 *)(a3 + 8), 50, (__int64)v57, 2);
    v21 = *(_DWORD *)(a2 + 44);
    if ( (v21 & 0x80u) == 0 )
      goto LABEL_37;
    goto LABEL_60;
  }
LABEL_58:
  sub_39A34D0(a1, a3, 135);
  v21 = *(_DWORD *)(a2 + 44);
  v22 = v21 & 3;
  if ( v22 == 2 )
    goto LABEL_59;
LABEL_33:
  if ( v22 == 1 )
  {
    v57[0] = 65547;
    sub_39A3560(a1, (__int64 *)(a3 + 8), 50, (__int64)v57, 3);
    v21 = *(_DWORD *)(a2 + 44);
  }
  else if ( v22 == 3 )
  {
    v57[0] = 65547;
    sub_39A3560(a1, (__int64 *)(a3 + 8), 50, (__int64)v57, 1);
    v21 = *(_DWORD *)(a2 + 44);
  }
  if ( (v21 & 0x80u) == 0 )
    goto LABEL_37;
LABEL_60:
  sub_39A34D0(a1, a3, 99);
  v21 = *(_DWORD *)(a2 + 44);
LABEL_37:
  if ( (v21 & 0x200000) != 0 )
    sub_39A34D0(a1, a3, 106);
}
