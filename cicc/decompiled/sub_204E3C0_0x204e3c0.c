// Function: sub_204E3C0
// Address: 0x204e3c0
//
__int64 __fastcall sub_204E3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6, unsigned int *a7)
{
  __int64 v7; // r14
  __int64 (__fastcall *v8)(__int64, __int64, __int64, __int64, __int64); // r9
  __int64 *v9; // r13
  __int64 result; // rax
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 (__fastcall *v15)(__int64, __int64, __int64, __int64, __int64); // rax
  int v16; // r8d
  int v17; // r12d
  __int64 v18; // rax
  int v19; // r14d
  int v20; // ebx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  bool v28; // al
  __int64 v29; // rcx
  __int64 v30; // rax
  char v31; // di
  char v32; // al
  bool v33; // al
  unsigned int v34; // eax
  bool v35; // zf
  int v36; // [rsp-10h] [rbp-110h]
  __int64 v37; // [rsp+0h] [rbp-100h]
  __int64 v38; // [rsp+8h] [rbp-F8h]
  __int64 v39; // [rsp+10h] [rbp-F0h]
  __int64 v40; // [rsp+18h] [rbp-E8h]
  const void *v41; // [rsp+20h] [rbp-E0h]
  const void *v42; // [rsp+28h] [rbp-D8h]
  __int64 v43; // [rsp+30h] [rbp-D0h]
  __int64 v44; // [rsp+30h] [rbp-D0h]
  const void *v45; // [rsp+38h] [rbp-C8h]
  __int64 *v46; // [rsp+40h] [rbp-C0h]
  __int64 v47; // [rsp+58h] [rbp-A8h]
  __int64 v48; // [rsp+58h] [rbp-A8h]
  __int64 v49; // [rsp+58h] [rbp-A8h]
  __int64 v50; // [rsp+58h] [rbp-A8h]
  __int64 v51; // [rsp+58h] [rbp-A8h]
  __int64 v52; // [rsp+58h] [rbp-A8h]
  __int64 v53; // [rsp+58h] [rbp-A8h]
  unsigned __int8 v55; // [rsp+60h] [rbp-A0h]
  int v56; // [rsp+6Ch] [rbp-94h]
  int v57; // [rsp+6Ch] [rbp-94h]
  unsigned __int8 v58; // [rsp+7Bh] [rbp-85h] BYREF
  unsigned int v59; // [rsp+7Ch] [rbp-84h] BYREF
  __int64 v60; // [rsp+80h] [rbp-80h] BYREF
  __int64 v61; // [rsp+88h] [rbp-78h]
  __int64 v62; // [rsp+90h] [rbp-70h] BYREF
  __int64 v63; // [rsp+98h] [rbp-68h]
  __int64 v64; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v65; // [rsp+A8h] [rbp-58h]
  char v66[8]; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v67; // [rsp+B8h] [rbp-48h]
  __int64 v68; // [rsp+C0h] [rbp-40h]

  v7 = a3;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  v41 = (const void *)(a1 + 96);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 104) = a1 + 120;
  v45 = (const void *)(a1 + 120);
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(_QWORD *)(a1 + 112) = 0x400000000LL;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 144) = 0x400000000LL;
  *(_BYTE *)(a1 + 172) = 0;
  v42 = (const void *)(a1 + 152);
  sub_20C7CE0(a3, a4, a6, a1, 0, 0);
  if ( *((_BYTE *)a7 + 4) )
  {
    v35 = *(_BYTE *)(a1 + 172) == 0;
    *(_DWORD *)(a1 + 168) = *a7;
    if ( v35 )
      *(_BYTE *)(a1 + 172) = 1;
  }
  else if ( *(_BYTE *)(a1 + 172) )
  {
    *(_BYTE *)(a1 + 172) = 0;
  }
  v9 = *(__int64 **)a1;
  result = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  v46 = (__int64 *)result;
  if ( result != *(_QWORD *)a1 )
  {
    do
    {
      v12 = *v9;
      v13 = v9[1];
      if ( *(_BYTE *)(a1 + 172) )
      {
        v14 = *(_QWORD *)v7;
        v8 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v7 + 392LL);
        if ( v8 != sub_1F42F80 )
        {
          v56 = v8(v7, a2, *a7, (unsigned int)v12, v9[1]);
          v32 = *(_BYTE *)(a1 + 172);
          goto LABEL_46;
        }
        LOBYTE(v60) = *v9;
        v61 = v13;
        if ( (_BYTE)v12 )
        {
          v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(v14 + 384);
          v56 = *(unsigned __int8 *)(v7 + (unsigned __int8)v12 + 1040);
          if ( v15 != sub_1F42DB0 )
          {
LABEL_48:
            v16 = v15(v7, a2, *a7, (unsigned int)v12, v13);
            goto LABEL_11;
          }
LABEL_9:
          LOBYTE(v60) = v12;
          v61 = v13;
          if ( (_BYTE)v12 )
          {
            v16 = *(unsigned __int8 *)(v7 + (unsigned __int8)v12 + 1155);
            goto LABEL_11;
          }
          if ( sub_1F58D20((__int64)&v60) )
          {
LABEL_54:
            v66[0] = 0;
            v67 = 0;
            LOBYTE(v62) = 0;
            sub_1F426C0(v7, a2, (unsigned int)v60, v13, (__int64)v66, (unsigned int *)&v64, &v62);
            LODWORD(v8) = v36;
            v16 = (unsigned __int8)v62;
            goto LABEL_11;
          }
          sub_1F40D10((__int64)v66, v7, a2, v60, v61);
          v22 = (unsigned __int8)v67;
          v24 = v68;
          LOBYTE(v62) = v67;
          v63 = v68;
          if ( (_BYTE)v67 )
            goto LABEL_32;
          if ( sub_1F58D20((__int64)&v62) )
          {
            v66[0] = 0;
            LOBYTE(v59) = 0;
            v67 = 0;
            sub_1F426C0(v7, a2, (unsigned int)v62, v24, (__int64)v66, (unsigned int *)&v64, &v59);
            goto LABEL_57;
          }
          sub_1F40D10((__int64)v66, v7, a2, v62, v63);
          v22 = (unsigned __int8)v67;
          v25 = v68;
          LOBYTE(v64) = v67;
          v65 = v68;
          if ( (_BYTE)v67 )
          {
LABEL_32:
            v16 = *(unsigned __int8 *)(v7 + v22 + 1155);
            goto LABEL_11;
          }
          if ( !sub_1F58D20((__int64)&v64) )
          {
            sub_1F40D10((__int64)v66, v7, a2, v64, v65);
            v26 = v40;
            LOBYTE(v26) = v67;
            v40 = v26;
            v16 = sub_1D5E9F0(v7, a2, (unsigned int)v26, v68);
            goto LABEL_11;
          }
          v66[0] = 0;
          v67 = 0;
          v58 = 0;
          sub_1F426C0(v7, a2, (unsigned int)v64, v25, (__int64)v66, &v59, &v58);
          goto LABEL_64;
        }
        if ( sub_1F58D20((__int64)&v60) )
          goto LABEL_45;
        v57 = sub_1F58D40((__int64)&v60);
        v62 = v60;
        v43 = v60;
        v63 = v61;
        v50 = v61;
        if ( sub_1F58D20((__int64)&v62) )
        {
LABEL_58:
          v66[0] = 0;
          v67 = 0;
          LOBYTE(v59) = 0;
          sub_1F426C0(v7, a2, (unsigned int)v62, v63, (__int64)v66, (unsigned int *)&v64, &v59);
          v31 = v59;
          goto LABEL_59;
        }
        sub_1F40D10((__int64)v66, v7, a2, v43, v50);
        v27 = (unsigned __int8)v67;
        LOBYTE(v64) = v67;
        v65 = v68;
        if ( (_BYTE)v67 )
          goto LABEL_60;
        v51 = v68;
        v28 = sub_1F58D20((__int64)&v64);
        v29 = v51;
        if ( v28 )
          goto LABEL_66;
        sub_1F40D10((__int64)v66, v7, a2, v64, v65);
        v30 = v37;
        LOBYTE(v30) = v67;
        v37 = v30;
      }
      else
      {
        v60 = *v9;
        v61 = v13;
        if ( (_BYTE)v12 )
        {
          v56 = *(unsigned __int8 *)(v7 + (unsigned __int8)v12 + 1040);
          goto LABEL_23;
        }
        if ( sub_1F58D20((__int64)&v60) )
        {
LABEL_45:
          v66[0] = 0;
          v67 = 0;
          LOBYTE(v62) = 0;
          v56 = sub_1F426C0(v7, a2, (unsigned int)v60, v13, (__int64)v66, (unsigned int *)&v64, &v62);
          v32 = *(_BYTE *)(a1 + 172);
          goto LABEL_46;
        }
        v57 = sub_1F58D40((__int64)&v60);
        v62 = v60;
        v44 = v60;
        v63 = v61;
        v52 = v61;
        if ( sub_1F58D20((__int64)&v62) )
          goto LABEL_58;
        sub_1F40D10((__int64)v66, v7, a2, v44, v52);
        v27 = (unsigned __int8)v67;
        LOBYTE(v64) = v67;
        v65 = v68;
        if ( (_BYTE)v67 )
        {
LABEL_60:
          v31 = *(_BYTE *)(v7 + v27 + 1155);
          goto LABEL_59;
        }
        v53 = v68;
        v33 = sub_1F58D20((__int64)&v64);
        v29 = v53;
        if ( v33 )
        {
LABEL_66:
          v66[0] = 0;
          v67 = 0;
          v58 = 0;
          sub_1F426C0(v7, a2, (unsigned int)v64, v29, (__int64)v66, &v59, &v58);
          v31 = v58;
          goto LABEL_59;
        }
        sub_1F40D10((__int64)v66, v7, a2, v64, v65);
        v30 = v38;
        LOBYTE(v30) = v67;
        v38 = v30;
      }
      v31 = sub_1D5E9F0(v7, a2, (unsigned int)v30, v68);
LABEL_59:
      v34 = sub_2045180(v31);
      v56 = (v34 + v57 - 1) / v34;
      v32 = *(_BYTE *)(a1 + 172);
LABEL_46:
      if ( v32 )
      {
        v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v7 + 384LL);
        if ( v15 != sub_1F42DB0 )
          goto LABEL_48;
        goto LABEL_9;
      }
LABEL_23:
      v22 = (unsigned __int8)v12;
      v60 = v12;
      v61 = v13;
      if ( (_BYTE)v12 )
        goto LABEL_32;
      if ( sub_1F58D20((__int64)&v60) )
        goto LABEL_54;
      sub_1F40D10((__int64)v66, v7, a2, v60, v61);
      v22 = (unsigned __int8)v67;
      LOBYTE(v62) = v67;
      v63 = v68;
      if ( (_BYTE)v67 )
        goto LABEL_32;
      v48 = v68;
      if ( sub_1F58D20((__int64)&v62) )
      {
        v66[0] = 0;
        v67 = 0;
        LOBYTE(v59) = 0;
        sub_1F426C0(v7, a2, (unsigned int)v62, v48, (__int64)v66, (unsigned int *)&v64, &v59);
LABEL_57:
        v16 = (unsigned __int8)v59;
LABEL_11:
        if ( v56 )
          goto LABEL_12;
        goto LABEL_30;
      }
      sub_1F40D10((__int64)v66, v7, a2, v62, v63);
      v22 = (unsigned __int8)v67;
      LOBYTE(v64) = v67;
      v65 = v68;
      if ( (_BYTE)v67 )
        goto LABEL_32;
      v49 = v68;
      if ( sub_1F58D20((__int64)&v64) )
      {
        v66[0] = 0;
        v67 = 0;
        v58 = 0;
        sub_1F426C0(v7, a2, (unsigned int)v64, v49, (__int64)v66, &v59, &v58);
LABEL_64:
        v16 = v58;
        goto LABEL_11;
      }
      sub_1F40D10((__int64)v66, v7, a2, v64, v65);
      v23 = v39;
      LOBYTE(v23) = v67;
      v39 = v23;
      v16 = sub_1D5E9F0(v7, a2, (unsigned int)v23, v68);
      if ( v56 )
      {
LABEL_12:
        v17 = a5;
        v18 = *(unsigned int *)(a1 + 112);
        v47 = v7;
        v19 = v16;
        v20 = a5 + v56;
        do
        {
          if ( *(_DWORD *)(a1 + 116) <= (unsigned int)v18 )
          {
            sub_16CD150(a1 + 104, v45, 0, 4, v16, (int)v8);
            v18 = *(unsigned int *)(a1 + 112);
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 104) + 4 * v18) = v17++;
          v18 = (unsigned int)(*(_DWORD *)(a1 + 112) + 1);
          *(_DWORD *)(a1 + 112) = v18;
        }
        while ( v17 != v20 );
        v16 = v19;
        v21 = *(unsigned int *)(a1 + 88);
        v7 = v47;
        if ( (unsigned int)v21 < *(_DWORD *)(a1 + 92) )
          goto LABEL_17;
LABEL_31:
        v55 = v16;
        sub_16CD150(a1 + 80, v41, 0, 1, v16, (int)v8);
        v21 = *(unsigned int *)(a1 + 88);
        v16 = v55;
        goto LABEL_17;
      }
LABEL_30:
      v20 = a5;
      v21 = *(unsigned int *)(a1 + 88);
      if ( (unsigned int)v21 >= *(_DWORD *)(a1 + 92) )
        goto LABEL_31;
LABEL_17:
      *(_BYTE *)(*(_QWORD *)(a1 + 80) + v21) = v16;
      result = *(unsigned int *)(a1 + 144);
      ++*(_DWORD *)(a1 + 88);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 148) )
      {
        sub_16CD150(a1 + 136, v42, 0, 4, v16, (int)v8);
        result = *(unsigned int *)(a1 + 144);
      }
      a5 = v20;
      v9 += 2;
      *(_DWORD *)(*(_QWORD *)(a1 + 136) + 4 * result) = v56;
      ++*(_DWORD *)(a1 + 144);
    }
    while ( v46 != v9 );
  }
  return result;
}
