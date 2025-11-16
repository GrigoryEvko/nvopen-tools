// Function: sub_37D3B80
// Address: 0x37d3b80
//
__int64 __fastcall sub_37D3B80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned int v10; // ebx
  unsigned int v11; // r12d
  unsigned int v12; // ebx
  unsigned int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  _DWORD *v18; // rax
  unsigned int v19; // eax
  int v20; // edx
  __int64 result; // rax
  char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rcx
  char *v26; // r14
  unsigned __int16 *v27; // rbx
  int v28; // edx
  _WORD *v29; // rdx
  _WORD *v30; // r13
  _WORD *v31; // rbx
  unsigned int v32; // esi
  int v33; // edx
  __int64 v34; // r9
  _DWORD *v35; // rdi
  int v36; // r11d
  unsigned int v37; // ecx
  _DWORD *v38; // rax
  int v39; // r8d
  _WORD *v40; // rax
  __int16 v41; // dx
  int v42; // eax
  int v43; // edx
  int v44; // eax
  __int64 v45; // [rsp+0h] [rbp-90h]
  __int64 v46; // [rsp+8h] [rbp-88h]
  __int64 v47; // [rsp+10h] [rbp-80h]
  __int64 v48; // [rsp+18h] [rbp-78h]
  __int64 v49; // [rsp+28h] [rbp-68h] BYREF
  _DWORD *v50; // [rsp+30h] [rbp-60h] BYREF
  char v51; // [rsp+38h] [rbp-58h]

  v5 = a1;
  v6 = a4;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 8) = a3;
  v8 = unk_5051170;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 48) = v8;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 120) = 0x800000000LL;
  *(_QWORD *)(a1 + 184) = a1 + 168;
  *(_QWORD *)(a1 + 192) = a1 + 168;
  *(_QWORD *)(a1 + 232) = a1 + 216;
  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 240) = a1 + 216;
  *(_QWORD *)(a1 + 296) = a1 + 312;
  *(_QWORD *)(a1 + 304) = 0x2000000000LL;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 824) = 0;
  *(_QWORD *)(a1 + 856) = 0;
  v9 = *(unsigned int *)(a4 + 16);
  v48 = a1 + 824;
  *(_DWORD *)(a1 + 280) = -1;
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  *(_DWORD *)(a1 + 848) = 0;
  v45 = a1 + 856;
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_DWORD *)(a1 + 880) = 0;
  *(_DWORD *)(a1 + 284) = v9;
  LODWORD(v50) = -1;
  if ( v9 )
  {
    sub_37BD740((unsigned __int64 *)(a1 + 64), 0, v9, (int *)&v50);
    v10 = *(_DWORD *)(a5 + 104);
    if ( !v10 )
      goto LABEL_3;
  }
  else
  {
    v10 = *(_DWORD *)(a5 + 104);
    if ( !v10 )
      goto LABEL_3;
  }
  sub_37BA440(a1, v10);
  v22 = sub_E922F0((_QWORD *)v6, v10);
  v25 = (__int64)&v22[2 * v23];
  if ( v22 != (char *)v25 )
  {
    v46 = v6;
    v26 = v22;
    v27 = (unsigned __int16 *)&v22[2 * v23];
    do
    {
      v28 = *(unsigned __int16 *)v26;
      v26 += 2;
      LODWORD(v49) = v28;
      sub_3361470((__int64)&v50, a1 + 112, (unsigned int *)&v49, v25, v24);
    }
    while ( v27 != (unsigned __int16 *)v26 );
    v5 = a1;
    v6 = v46;
  }
LABEL_3:
  v49 = 8;
  sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
  v49 = 0x100000010LL;
  sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
  v49 = 0x200000020LL;
  sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
  v49 = 0x300000040LL;
  sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
  v49 = 0x400000080LL;
  sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
  v49 = 0x500000100LL;
  sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
  v49 = 0x600000200LL;
  sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
  if ( *(_DWORD *)(v6 + 96) > 1u )
  {
    v11 = 1;
    do
    {
      v12 = sub_2FF7530(v6, v11);
      v13 = sub_2FF7550(v6, v11);
      if ( v13 <= 0xEA60 && v12 <= 0xEA60 )
      {
        v14 = *(_DWORD *)(v5 + 840);
        LOWORD(v49) = v12;
        WORD1(v49) = v13;
        HIDWORD(v49) = v14;
        sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
      }
      ++v11;
    }
    while ( v11 < *(_DWORD *)(v6 + 96) );
  }
  v15 = *(_QWORD *)(v6 + 288);
  v16 = *(_QWORD *)(v6 + 280);
  v47 = v15;
  if ( v15 != v16 )
  {
    v17 = *(_QWORD *)(v6 + 280);
    while ( 1 )
    {
      v18 = (_DWORD *)*(unsigned int *)(*(_QWORD *)(v6 + 312)
                                      + 16LL
                                      * (*(unsigned __int16 *)(**(_QWORD **)v17 + 24LL)
                                       + *(_DWORD *)(v6 + 328) * (unsigned int)((v15 - v16) >> 3)));
      v51 = 0;
      v50 = v18;
      v19 = sub_CA1930(&v50);
      if ( v19 <= 0x200 )
      {
        v20 = *(_DWORD *)(v5 + 840);
        LODWORD(v49) = (unsigned __int16)v19;
        HIDWORD(v49) = v20;
        sub_37C5D80((__int64)&v50, v48, (unsigned __int16 *)&v49, (_DWORD *)&v49 + 1);
      }
      v17 += 8;
      if ( v47 == v17 )
        break;
      v15 = *(_QWORD *)(v6 + 288);
      v16 = *(_QWORD *)(v6 + 280);
    }
  }
  result = *(unsigned int *)(v5 + 840);
  if ( !(_DWORD)result )
    goto LABEL_16;
  v29 = *(_WORD **)(v5 + 832);
  v30 = &v29[4 * *(unsigned int *)(v5 + 848)];
  if ( v29 == v30 )
    goto LABEL_16;
  while ( 1 )
  {
    v31 = v29;
    if ( *v29 != 0xFFFF )
      break;
    if ( v29[1] != 0xFFFF )
      goto LABEL_25;
LABEL_56:
    v29 += 4;
    if ( v30 == v29 )
      goto LABEL_16;
  }
  if ( *v29 == 0xFFFE && v29[1] == 0xFFFE )
    goto LABEL_56;
LABEL_25:
  if ( v30 == v29 )
    goto LABEL_16;
LABEL_29:
  v32 = *(_DWORD *)(v5 + 880);
  if ( !v32 )
  {
    ++*(_QWORD *)(v5 + 856);
    v50 = 0;
LABEL_53:
    v32 *= 2;
    goto LABEL_54;
  }
  v33 = *((_DWORD *)v31 + 1);
  v34 = *(_QWORD *)(v5 + 864);
  v35 = 0;
  v36 = 1;
  v37 = (v32 - 1) & (37 * v33);
  v38 = (_DWORD *)(v34 + 8LL * v37);
  v39 = *v38;
  if ( v33 == *v38 )
  {
LABEL_31:
    v40 = v38 + 1;
    goto LABEL_32;
  }
  while ( v39 != -1 )
  {
    if ( !v35 && v39 == -2 )
      v35 = v38;
    v37 = (v32 - 1) & (v36 + v37);
    v38 = (_DWORD *)(v34 + 8LL * v37);
    v39 = *v38;
    if ( v33 == *v38 )
      goto LABEL_31;
    ++v36;
  }
  if ( !v35 )
    v35 = v38;
  v42 = *(_DWORD *)(v5 + 872);
  ++*(_QWORD *)(v5 + 856);
  v43 = v42 + 1;
  v50 = v35;
  if ( 4 * (v42 + 1) >= 3 * v32 )
    goto LABEL_53;
  if ( v32 - *(_DWORD *)(v5 + 876) - v43 <= v32 >> 3 )
  {
LABEL_54:
    sub_37C67C0(v45, v32);
    sub_37BDC70(v45, (int *)v31 + 1, &v50);
    v35 = v50;
    v43 = *(_DWORD *)(v5 + 872) + 1;
  }
  *(_DWORD *)(v5 + 872) = v43;
  if ( *v35 != -1 )
    --*(_DWORD *)(v5 + 876);
  v44 = *((_DWORD *)v31 + 1);
  v35[1] = 0;
  *v35 = v44;
  v40 = v35 + 1;
LABEL_32:
  v41 = *v31;
  v31 += 4;
  *v40 = v41;
  v40[1] = *(v31 - 3);
  while ( v30 != v31 )
  {
    if ( *v31 == 0xFFFF )
    {
      if ( v31[1] != 0xFFFF )
        goto LABEL_28;
    }
    else if ( *v31 != 0xFFFE || v31[1] != 0xFFFE )
    {
LABEL_28:
      if ( v31 == v30 )
        break;
      goto LABEL_29;
    }
    v31 += 4;
  }
  result = *(unsigned int *)(v5 + 840);
LABEL_16:
  *(_DWORD *)(v5 + 288) = result;
  return result;
}
