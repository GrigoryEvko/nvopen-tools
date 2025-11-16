// Function: sub_2AB6F90
// Address: 0x2ab6f90
//
__int64 __fastcall sub_2AB6F90(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  char v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // rsi
  int v13; // eax
  bool v14; // zf
  int v15; // r14d
  unsigned int v16; // eax
  __int64 v17; // r8
  __int64 *v18; // rbx
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int8 *v31; // rax
  __int64 v32; // r8
  __int64 v33; // rdi
  __int64 v34; // rax
  int v35; // r13d
  int v36; // eax
  __int64 v37; // r9
  __int64 v38; // r15
  __int64 v39; // r9
  char *v40; // rax
  __int64 v41; // rax
  int v42; // r13d
  __int64 v43; // r14
  __int64 v44; // r9
  bool v45; // al
  __int64 v46; // [rsp+8h] [rbp-A8h]
  bool v47; // [rsp+1Eh] [rbp-92h]
  bool v48; // [rsp+1Fh] [rbp-91h]
  __int64 v49; // [rsp+20h] [rbp-90h]
  __int64 v50; // [rsp+28h] [rbp-88h]
  __int64 v51; // [rsp+28h] [rbp-88h]
  __int64 v52; // [rsp+28h] [rbp-88h]
  __int64 v53; // [rsp+28h] [rbp-88h]
  __int64 v54; // [rsp+28h] [rbp-88h]
  __int64 v56; // [rsp+30h] [rbp-80h]
  __int64 v57; // [rsp+30h] [rbp-80h]
  __int64 v58; // [rsp+30h] [rbp-80h]
  __int64 v59; // [rsp+30h] [rbp-80h]
  __int64 v60; // [rsp+30h] [rbp-80h]
  __int64 v61; // [rsp+38h] [rbp-78h] BYREF
  __int64 v62; // [rsp+48h] [rbp-68h] BYREF
  __int64 v63; // [rsp+50h] [rbp-60h] BYREF
  __int64 v64; // [rsp+58h] [rbp-58h] BYREF
  __int64 v65; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v66; // [rsp+68h] [rbp-48h]
  __int64 (__fastcall *v67)(const __m128i **, const __m128i *, int); // [rsp+70h] [rbp-40h]
  char (__fastcall *v68)(_QWORD **, __int64); // [rsp+78h] [rbp-38h]

  v5 = 0;
  v66 = &v61;
  v61 = a2;
  v65 = a1;
  v68 = sub_2AB2E80;
  v67 = sub_2AA7C60;
  v8 = sub_2BF1270(&v65, a5);
  sub_A17130((__int64)&v65);
  if ( !v8 )
    return v5;
  v49 = 0;
  if ( (unsigned __int8)sub_B19060(*(_QWORD *)(a1 + 32) + 440LL, v61, v9, v10) )
    v49 = sub_2AB6F10(a1, *(_QWORD *)(v61 + 40));
  v11 = v61;
  v12 = v61;
  v13 = sub_2AAA2B0(*(_QWORD *)(a1 + 40), v61, *(_DWORD *)a5, *(_BYTE *)(a5 + 4));
  v14 = v13 == 2;
  v15 = v13;
  v16 = v13 - 1;
  v48 = v14;
  v47 = v16 <= 1;
  if ( *(_BYTE *)v11 == 61 )
  {
    v17 = *a3;
    if ( v16 > 1 )
      goto LABEL_17;
  }
  else
  {
    v17 = a3[1];
    if ( v16 > 1 )
    {
LABEL_6:
      v18 = (__int64 *)*a3;
      v62 = *(_QWORD *)(v11 + 48);
      if ( v62 )
      {
        v56 = v17;
        sub_2AAAFA0(&v62);
        v17 = v56;
      }
      v57 = v17;
      v5 = sub_22077B0(0x70u);
      if ( v5 )
      {
        v63 = v62;
        if ( v62 )
        {
          sub_2AAAFA0(&v63);
          v66 = v18;
          v65 = v57;
          v64 = v63;
          if ( v63 )
            sub_2AAAFA0(&v64);
        }
        else
        {
          v65 = v57;
          v66 = v18;
          v64 = 0;
        }
        sub_2AAF310(v5, 22, &v65, 2, &v64, v19);
        sub_9C6650(&v64);
        *(_QWORD *)(v5 + 96) = v11;
        *(_BYTE *)(v5 + 106) = 0;
        *(_QWORD *)(v5 + 40) = &unk_4A24740;
        *(_QWORD *)v5 = &unk_4A24708;
        *(_BYTE *)(v5 + 104) = v47;
        *(_BYTE *)(v5 + 105) = v48;
        sub_9C6650(&v63);
        *(_QWORD *)v5 = &unk_4A248A8;
        *(_QWORD *)(v5 + 40) = &unk_4A248E8;
        if ( v49 )
        {
          sub_2AAECA0(v5 + 40, v49, (__int64)&unk_4A248A8, v20, v21, v22);
          *(_BYTE *)(v5 + 106) = 1;
        }
      }
      sub_9C6650(&v62);
      return v5;
    }
  }
  v50 = v17;
  v31 = sub_BD3990(*(unsigned __int8 **)(v17 + 40), v12);
  v32 = v50;
  v33 = (__int64)v31;
  if ( *v31 != 63 )
    v33 = 0;
  if ( v15 == 2 )
  {
    v41 = *(_QWORD *)(a1 + 40);
    if ( *(_BYTE *)(v41 + 108) && *(_DWORD *)(v41 + 100) || !v33 || (v45 = sub_B4DE30(v33), v32 = v50, !v45) )
      v42 = 0;
    else
      v42 = 3;
    v43 = *(_QWORD *)a1 + 272LL;
    if ( *(_BYTE *)v61 == 61 )
      v46 = *(_QWORD *)(v61 + 8);
    else
      v46 = *(_QWORD *)(*(_QWORD *)(v61 - 64) + 8LL);
    v62 = *(_QWORD *)(v61 + 48);
    if ( v62 )
    {
      v53 = v32;
      sub_2AAAFA0(&v62);
      v32 = v53;
    }
    v54 = v32;
    v38 = sub_22077B0(0xA8u);
    if ( v38 )
    {
      v63 = v62;
      if ( v62 )
      {
        sub_2AAAFA0(&v63);
        v66 = (__int64 *)v43;
        v65 = v54;
        v64 = v63;
        if ( v63 )
          sub_2AAAFA0(&v64);
      }
      else
      {
        v66 = (__int64 *)v43;
        v65 = v54;
        v64 = 0;
      }
      sub_2AAF4A0(v38, 13, &v65, 2, &v64, v44);
      sub_9C6650(&v64);
      *(_BYTE *)(v38 + 152) = 4;
      *(_DWORD *)(v38 + 156) = v42;
      *(_QWORD *)v38 = &unk_4A23258;
      *(_QWORD *)(v38 + 96) = &unk_4A232C8;
      *(_QWORD *)(v38 + 40) = &unk_4A23290;
      sub_9C6650(&v63);
      v40 = (char *)&unk_4A24250;
      goto LABEL_45;
    }
  }
  else
  {
    v34 = v61;
    if ( *(_BYTE *)v61 == 61 )
      v46 = *(_QWORD *)(v61 + 8);
    else
      v46 = *(_QWORD *)(*(_QWORD *)(v61 - 64) + 8LL);
    v35 = 0;
    if ( v33 )
    {
      v36 = sub_B4DE20(v33);
      v32 = v50;
      v35 = v36;
      v34 = v61;
    }
    v62 = *(_QWORD *)(v34 + 48);
    if ( v62 )
    {
      v51 = v32;
      sub_2AAAFA0(&v62);
      v32 = v51;
    }
    v52 = v32;
    v38 = sub_22077B0(0xA8u);
    if ( v38 )
    {
      v63 = v52;
      v64 = v62;
      if ( v62 )
      {
        sub_2AAAFA0(&v64);
        v65 = v64;
        if ( v64 )
          sub_2AAAFA0(&v65);
        sub_2AAF4A0(v38, 12, &v63, 1, &v65, v39);
      }
      else
      {
        v65 = 0;
        sub_2AAF4A0(v38, 12, &v63, 1, &v65, v37);
      }
      sub_9C6650(&v65);
      *(_BYTE *)(v38 + 152) = 4;
      *(_DWORD *)(v38 + 156) = v35;
      *(_QWORD *)v38 = &unk_4A23258;
      *(_QWORD *)(v38 + 96) = &unk_4A232C8;
      *(_QWORD *)(v38 + 40) = &unk_4A23290;
      sub_9C6650(&v64);
      v40 = (char *)&unk_4A242F0;
LABEL_45:
      *(_QWORD *)v38 = v40 + 16;
      *(_QWORD *)(v38 + 40) = v40 + 88;
      *(_QWORD *)(v38 + 96) = v40 + 144;
      *(_QWORD *)(v38 + 160) = v46;
    }
  }
  sub_9C6650(&v62);
  sub_2AAFF40(**(_QWORD **)(a1 + 56), (_QWORD *)v38, *(unsigned __int64 **)(*(_QWORD *)(a1 + 56) + 8LL));
  if ( v38 )
    v17 = v38 + 96;
  else
    v17 = 0;
  v11 = v61;
  if ( *(_BYTE *)v61 != 61 )
    goto LABEL_6;
LABEL_17:
  v63 = *(_QWORD *)(v11 + 48);
  if ( v63 )
  {
    v58 = v17;
    sub_2AAAFA0(&v63);
    v17 = v58;
  }
  v59 = v17;
  v5 = sub_22077B0(0xA8u);
  if ( v5 )
  {
    v25 = v59;
    v64 = v63;
    if ( v63 )
    {
      sub_2AAAFA0(&v64);
      v25 = v59;
      v65 = v64;
      if ( v64 )
      {
        sub_2AAAFA0(&v65);
        v25 = v59;
      }
    }
    else
    {
      v65 = 0;
    }
    *(_BYTE *)(v5 + 8) = 20;
    *(_QWORD *)(v5 + 24) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)v5 = &unk_4A231A8;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 64) = v25;
    *(_QWORD *)(v5 + 40) = &unk_4A23170;
    *(_QWORD *)(v5 + 48) = v5 + 64;
    *(_QWORD *)(v5 + 56) = 0x200000001LL;
    v26 = *(unsigned int *)(v25 + 24);
    if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v25 + 28) )
    {
      v60 = v25;
      sub_C8D5F0(v25 + 16, (const void *)(v25 + 32), v26 + 1, 8u, v25, v24);
      v25 = v60;
      v26 = *(unsigned int *)(v60 + 24);
    }
    *(_QWORD *)(*(_QWORD *)(v25 + 16) + 8 * v26) = v5 + 40;
    ++*(_DWORD *)(v25 + 24);
    *(_QWORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 40) = &unk_4A23AA8;
    v27 = v65;
    *(_QWORD *)v5 = &unk_4A23A70;
    *(_QWORD *)(v5 + 88) = v27;
    if ( v27 )
      sub_2AAAFA0((__int64 *)(v5 + 88));
    sub_9C6650(&v65);
    *(_QWORD *)(v5 + 96) = v11;
    *(_BYTE *)(v5 + 106) = 0;
    *(_QWORD *)(v5 + 40) = &unk_4A24740;
    *(_QWORD *)v5 = &unk_4A24708;
    *(_BYTE *)(v5 + 104) = v47;
    *(_BYTE *)(v5 + 105) = v48;
    sub_9C6650(&v64);
    sub_2BF0340(v5 + 112, 1, v11, v5);
    *(_QWORD *)v5 = &unk_4A24778;
    *(_QWORD *)(v5 + 112) = &unk_4A247F0;
    *(_QWORD *)(v5 + 40) = &unk_4A247B8;
    if ( v49 )
    {
      sub_2AAECA0(v5 + 40, v49, (__int64)&unk_4A247B8, v28, v29, v30);
      *(_BYTE *)(v5 + 106) = 1;
    }
  }
  sub_9C6650(&v63);
  return v5;
}
