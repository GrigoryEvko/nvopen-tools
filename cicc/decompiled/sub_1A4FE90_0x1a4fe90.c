// Function: sub_1A4FE90
// Address: 0x1a4fe90
//
__int64 *__fastcall sub_1A4FE90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 *result; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r9
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rsi
  __int64 v28; // r8
  __int64 v29; // r12
  __int64 v30; // r13
  unsigned int v31; // ebx
  __int64 v32; // r15
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rcx
  int v36; // eax
  __int64 v37; // rax
  int v38; // edx
  __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // rdi
  unsigned __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  int v50; // eax
  __int64 v51; // rax
  int v52; // edx
  __int64 v53; // rdx
  __int64 *v54; // rax
  __int64 v55; // rcx
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // r13
  __int64 v60; // [rsp+18h] [rbp-98h]
  __int64 v62; // [rsp+28h] [rbp-88h]
  __int64 v63; // [rsp+30h] [rbp-80h]
  __int64 v64; // [rsp+30h] [rbp-80h]
  __int64 v65; // [rsp+38h] [rbp-78h]
  __int64 v66; // [rsp+38h] [rbp-78h]
  _QWORD v69[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v70[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v71; // [rsp+70h] [rbp-40h]

  v14 = *(_QWORD *)(a2 + 48);
  v15 = v14 - 24;
  if ( !v14 )
    v15 = 0;
  v62 = v15;
  v16 = sub_157F280(a1);
  v60 = v17;
  result = v70;
  if ( v16 != v17 )
  {
    while ( 1 )
    {
      v69[0] = sub_1649960(v16);
      v71 = 773;
      v69[1] = v19;
      v70[0] = (__int64)v69;
      v70[1] = (__int64)".split";
      v20 = *(_QWORD *)v16;
      v21 = sub_1648B60(64);
      v25 = v21;
      if ( v21 )
      {
        v26 = v21;
        sub_15F1EA0(v21, v20, 53, 0, 0, v62);
        *(_DWORD *)(v25 + 56) = 2;
        sub_164B780(v25, v70);
        sub_1648880(v25, *(_DWORD *)(v25 + 56), 1);
      }
      else
      {
        v26 = 0;
      }
      v27 = *(_DWORD *)(v16 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v27 )
        break;
LABEL_27:
      sub_164D160(v16, v25, a6, a7, a8, a9, v23, v24, a12, a13);
      v50 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
      if ( v50 == *(_DWORD *)(v25 + 56) )
      {
        sub_15F55D0(v25, v25, v46, v47, v48, v49);
        v50 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
      }
      v51 = (v50 + 1) & 0xFFFFFFF;
      v52 = v51 | *(_DWORD *)(v25 + 20) & 0xF0000000;
      *(_DWORD *)(v25 + 20) = v52;
      if ( (v52 & 0x40000000) != 0 )
        v53 = *(_QWORD *)(v25 - 8);
      else
        v53 = v26 - 24 * v51;
      v54 = (__int64 *)(v53 + 24LL * (unsigned int)(v51 - 1));
      if ( *v54 )
      {
        v55 = v54[1];
        v56 = v54[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v56 = v55;
        if ( v55 )
          *(_QWORD *)(v55 + 16) = *(_QWORD *)(v55 + 16) & 3LL | v56;
      }
      *v54 = v16;
      if ( v16 )
      {
        v57 = *(_QWORD *)(v16 + 8);
        v54[1] = v57;
        if ( v57 )
          *(_QWORD *)(v57 + 16) = (unsigned __int64)(v54 + 1) | *(_QWORD *)(v57 + 16) & 3LL;
        v54[2] = (v16 + 8) | v54[2] & 3;
        *(_QWORD *)(v16 + 8) = v54;
      }
      v58 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
        v59 = *(_QWORD *)(v25 - 8);
      else
        v59 = v26 - 24 * v58;
      *(_QWORD *)(v59 + 8LL * (unsigned int)(v58 - 1) + 24LL * *(unsigned int *)(v25 + 56) + 8) = a1;
      result = *(__int64 **)(v16 + 32);
      if ( !result )
        BUG();
      v16 = 0;
      if ( *((_BYTE *)result - 8) == 77 )
        v16 = (__int64)(result - 3);
      if ( v60 == v16 )
        return result;
    }
    v28 = a3;
    v29 = v26;
    v30 = v25;
    v31 = v27 - 1;
    v32 = 8LL * ((int)v27 - 1);
    while ( 1 )
    {
      while ( 1 )
      {
        v34 = *(unsigned int *)(v16 + 56);
        if ( (*(_BYTE *)(v16 + 23) & 0x40) == 0 )
          break;
        v33 = *(_QWORD *)(v16 - 8);
        if ( v28 == *(_QWORD *)(v32 + v33 + 24 * v34 + 8) )
          goto LABEL_12;
LABEL_9:
        --v31;
        v32 -= 8;
        if ( v31 == -1 )
          goto LABEL_26;
      }
      v33 = v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
      if ( v28 != *(_QWORD *)(v32 + v33 + 24 * v34 + 8) )
        goto LABEL_9;
LABEL_12:
      v35 = *(_QWORD *)(v33 + 3 * v32);
      if ( a5 )
      {
        v27 = v31;
        v63 = v28;
        v65 = *(_QWORD *)(v33 + 3 * v32);
        sub_15F5350(v16, v31, 1);
        v28 = v63;
        v35 = v65;
        v36 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
        if ( v36 == *(_DWORD *)(v30 + 56) )
        {
LABEL_48:
          v64 = v28;
          v66 = v35;
          sub_15F55D0(v30, v27, v33, v35, v28, v22);
          v28 = v64;
          v35 = v66;
          v36 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
        }
      }
      else
      {
        v36 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
        if ( v36 == *(_DWORD *)(v30 + 56) )
          goto LABEL_48;
      }
      v37 = (v36 + 1) & 0xFFFFFFF;
      v38 = v37 | *(_DWORD *)(v30 + 20) & 0xF0000000;
      *(_DWORD *)(v30 + 20) = v38;
      if ( (v38 & 0x40000000) != 0 )
        v39 = *(_QWORD *)(v30 - 8);
      else
        v39 = v29 - 24 * v37;
      v40 = (__int64 *)(v39 + 24LL * (unsigned int)(v37 - 1));
      if ( *v40 )
      {
        v41 = v40[1];
        v42 = v40[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v42 = v41;
        if ( v41 )
          *(_QWORD *)(v41 + 16) = *(_QWORD *)(v41 + 16) & 3LL | v42;
      }
      *v40 = v35;
      if ( v35 )
      {
        v43 = *(_QWORD *)(v35 + 8);
        v40[1] = v43;
        if ( v43 )
          *(_QWORD *)(v43 + 16) = (unsigned __int64)(v40 + 1) | *(_QWORD *)(v43 + 16) & 3LL;
        v40[2] = (v35 + 8) | v40[2] & 3;
        *(_QWORD *)(v35 + 8) = v40;
      }
      v44 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v30 + 23) & 0x40) != 0 )
        v45 = *(_QWORD *)(v30 - 8);
      else
        v45 = v29 - 24 * v44;
      v27 = a4;
      --v31;
      v32 -= 8;
      *(_QWORD *)(v45 + 8LL * (unsigned int)(v44 - 1) + 24LL * *(unsigned int *)(v30 + 56) + 8) = a4;
      if ( v31 == -1 )
      {
LABEL_26:
        v25 = v30;
        v26 = v29;
        a3 = v28;
        goto LABEL_27;
      }
    }
  }
  return result;
}
