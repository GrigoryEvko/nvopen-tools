// Function: sub_1DAC9A0
// Address: 0x1dac9a0
//
void __fastcall sub_1DAC9A0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r13
  unsigned int v7; // edx
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // r14d
  __int64 v11; // r15
  unsigned int v12; // ebx
  unsigned int v13; // eax
  bool v14; // bl
  int v15; // esi
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned __int64 *v18; // rcx
  __int64 v19; // rdx
  int v20; // esi
  __int64 v21; // rax
  __int64 v22; // r14
  unsigned __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rbx
  int v31; // r15d
  __int64 v32; // rax
  __int64 v33; // r14
  unsigned int v34; // eax
  unsigned int v35; // ecx
  __int64 v36; // rdx
  unsigned int v37; // r10d
  __int64 v38; // rax
  __int64 v39; // r8
  unsigned int v40; // r10d
  char v41; // dl
  unsigned __int64 v42; // rax
  int v43; // edx
  __int64 v44; // rcx
  __int64 v45; // rdi
  _QWORD *v46; // rax
  unsigned int v47; // r15d
  __int64 v48; // rbx
  unsigned int v49; // r9d
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rax
  unsigned __int64 *v54; // rax
  unsigned int i; // r14d
  __int64 v56; // rax
  unsigned int v57; // edx
  __int64 v58; // [rsp+10h] [rbp-B0h]
  unsigned int v59; // [rsp+18h] [rbp-A8h]
  unsigned int v60; // [rsp+18h] [rbp-A8h]
  __int64 v61; // [rsp+20h] [rbp-A0h]
  unsigned int v62; // [rsp+20h] [rbp-A0h]
  __int64 v63; // [rsp+20h] [rbp-A0h]
  __int64 v64; // [rsp+28h] [rbp-98h]
  unsigned int v65; // [rsp+28h] [rbp-98h]
  __int64 *v66; // [rsp+30h] [rbp-90h]
  unsigned int v67; // [rsp+30h] [rbp-90h]
  __int64 v68; // [rsp+38h] [rbp-88h]
  unsigned __int64 v69; // [rsp+38h] [rbp-88h]
  char v70; // [rsp+43h] [rbp-7Dh]
  unsigned int v73[4]; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v74[4]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v75[10]; // [rsp+70h] [rbp-50h] BYREF

  v4 = a1 + 8;
  v7 = *(_DWORD *)(a1 + 16);
  v68 = a2;
  if ( !v7 || (v8 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v8 + 12) >= *(_DWORD *)(v8 + 8)) )
  {
    v16 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 80LL);
    sub_3945E40(a1 + 8, *(unsigned int *)(*(_QWORD *)a1 + 80LL));
    ++*(_DWORD *)(*(_QWORD *)(a1 + 8) + v16 + 12);
    v7 = *(_DWORD *)(a1 + 16);
    v8 = *(_QWORD *)(a1 + 8);
  }
  v9 = v8 + 16LL * v7 - 16;
  v10 = *(_DWORD *)(v9 + 12);
  v11 = *(_QWORD *)v9;
  if ( !v10
    && (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) < (*(_DWORD *)((*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                                  + 24)
                                                                                      | (unsigned int)(*(__int64 *)v11 >> 1)
                                                                                      & 3) )
  {
    v21 = sub_3945DA0(v4, v7 - 1);
    if ( v21 )
    {
      v22 = v21 & 0x3F;
      v23 = v21 & 0xFFFFFFFFFFFFFFC0LL;
      v24 = *(unsigned int *)(a1 + 16);
      v9 = *(_QWORD *)(a1 + 8) + 16 * v24 - 16;
      v25 = (unsigned int)v22;
      v11 = *(_QWORD *)v9;
      if ( ((*(_DWORD *)(v23 + 4 * v22 + 144) ^ a4) & 0x7FFFFFFF) != 0
        || ((*(_BYTE *)(v23 + 4 * (v22 + 36) + 3) ^ HIBYTE(a4)) & 0x80u) != 0
        || (v26 = 16 * v22, *(_QWORD *)(v23 + v26 + 8) != a2) )
      {
        v10 = *(_DWORD *)(v9 + 12);
        goto LABEL_5;
      }
      v69 = v23;
      v61 = v25;
      v66 = (__int64 *)(v23 + v26 + 8);
      sub_3945E40(v4, (unsigned int)(v24 - 1));
      if ( (*(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a3 >> 1) & 3) <= (*(_DWORD *)((*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(*(__int64 *)v11 >> 1)
                                                                                           & 3) )
      {
        v29 = (__int64)v66;
        v27 = v61;
        if ( ((*(_DWORD *)(v11 + 144) ^ a4) & 0x7FFFFFFF) != 0
          || (v28 = HIBYTE(a4), ((*(_BYTE *)(v11 + 147) ^ HIBYTE(a4)) & 0x80u) != 0)
          || a3 != *(_QWORD *)v11 )
        {
          *(_QWORD *)(v69 + 16 * v61 + 8) = a3;
          v19 = *v66;
          v20 = *(_DWORD *)(a1 + 16) - 1;
          goto LABEL_12;
        }
      }
      v68 = *(_QWORD *)(v69 + v26);
      sub_1DAB250(a1, 0, v27, v28, v29);
    }
    else
    {
      **(_QWORD **)a1 = a2;
    }
    v9 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
    v10 = *(_DWORD *)(v9 + 12);
    v11 = *(_QWORD *)v9;
  }
LABEL_5:
  v12 = *(_DWORD *)(v9 + 8);
  v13 = sub_1DAA830(v11, (unsigned int *)(v9 + 12), v12, v68, a3, a4);
  v14 = v12 == v10;
  if ( v13 <= 9 )
    goto LABEL_6;
  v30 = (unsigned int)(*(_DWORD *)(a1 + 16) - 1);
  v31 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16 * v30 + 12);
  v32 = sub_3945DA0(v4, v30);
  v33 = v32;
  if ( v32 )
  {
    v67 = 2;
    v34 = (v32 & 0x3F) + 1;
    v75[0] = v33 & 0xFFFFFFFFFFFFFFC0LL;
    v35 = 1;
    v73[0] = v34;
    v31 += v34;
  }
  else
  {
    v67 = 1;
    v34 = 0;
    v35 = 0;
  }
  v36 = *(_QWORD *)(a1 + 8) + 16 * v30;
  v37 = *(_DWORD *)(v36 + 8);
  v59 = v35;
  v64 = v35;
  v73[v35] = v37;
  v62 = v34 + v37;
  v75[v35] = *(_QWORD *)v36;
  v38 = sub_3945FF0(v4, (unsigned int)v30);
  v39 = v64;
  v40 = v62;
  if ( v38 )
  {
    v41 = v38;
    v42 = v38 & 0xFFFFFFFFFFFFFFC0LL;
    v39 = v67;
    v65 = v59 + 2;
    v43 = (v41 & 0x3F) + 1;
    v40 = v43 + v62;
    v73[v67] = v43;
    v75[v67] = v42;
    if ( v43 + v62 + 1 > 9 * v59 + 18 )
      goto LABEL_26;
LABEL_46:
    v70 = 0;
    v67 = 0;
    goto LABEL_28;
  }
  if ( v62 + 1 <= 9 * (unsigned int)(v67 != 1) + 9 )
  {
    v65 = v67;
    goto LABEL_46;
  }
  if ( v67 == 1 )
  {
    v43 = v73[1];
    v42 = v75[1];
    v44 = 1;
    v65 = 2;
    v39 = 1;
    goto LABEL_27;
  }
  v43 = v73[v64];
  v42 = v75[v64];
  v67 = v59;
  v65 = 2;
LABEL_26:
  v44 = v65++;
LABEL_27:
  v75[v44] = v42;
  v60 = v40;
  v45 = *(_QWORD *)(*(_QWORD *)a1 + 88LL);
  v73[v44] = v43;
  v63 = v39;
  v73[v39] = 0;
  v46 = sub_1DA89E0(v45);
  v70 = 1;
  v40 = v60;
  v75[v63] = v46;
LABEL_28:
  v58 = sub_39461C0(v65, v40, 9, (unsigned int)v73, (unsigned int)v74, v31, 1);
  sub_1DABCA0((__int64)v75, v65, v73, v74);
  if ( v33 )
    sub_3945E40(v4, (unsigned int)v30);
  v47 = v30;
  v48 = 0;
  while ( 1 )
  {
    v49 = v74[v48];
    v50 = v49 - 1;
    v51 = v75[v48];
    v52 = *(_QWORD *)(v51 + 16 * v50 + 8);
    if ( v67 != (_DWORD)v48 || !v70 )
      break;
    ++v48;
    v47 += (unsigned __int8)sub_1DAC6C0(a1, v47, v50 | v51 & 0xFFFFFFFFFFFFFFC0LL, v52);
    if ( v65 == v48 )
      goto LABEL_42;
LABEL_34:
    sub_39460A0(v4, v47);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * v47 + 8) = v49;
  if ( v47 )
  {
    v53 = *(_QWORD *)(a1 + 8) + 16LL * (v47 - 1);
    v54 = (unsigned __int64 *)(*(_QWORD *)v53 + 8LL * *(unsigned int *)(v53 + 12));
    *v54 = v50 | *v54 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v48;
  sub_1DA99F0(a1, v47, v52);
  if ( v65 != v48 )
    goto LABEL_34;
LABEL_42:
  for ( i = v65 - 1; i != (_DWORD)v58; --i )
    sub_3945E40(v4, v47);
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * v47 + 12) = HIDWORD(v58);
  v56 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v57 = *(_DWORD *)(v56 + 8);
  v14 = *(_DWORD *)(v56 + 12) == v57;
  v13 = sub_1DAA830(*(_QWORD *)v56, (unsigned int *)(v56 + 12), v57, v68, a3, a4);
LABEL_6:
  v15 = *(_DWORD *)(a1 + 16);
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v15 - 1) + 8) = v13;
  if ( v15 == 1 )
  {
    if ( !v14 )
      return;
LABEL_11:
    v19 = a3;
    v20 = *(_DWORD *)(a1 + 16) - 1;
LABEL_12:
    sub_1DA99F0(a1, v20, v19);
    return;
  }
  v17 = *(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v15 - 2);
  v18 = (unsigned __int64 *)(*(_QWORD *)v17 + 8LL * *(unsigned int *)(v17 + 12));
  *v18 = *v18 & 0xFFFFFFFFFFFFFFC0LL | (v13 - 1);
  if ( v14 )
    goto LABEL_11;
}
