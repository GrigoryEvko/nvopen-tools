// Function: sub_3109500
// Address: 0x3109500
//
__int64 *__fastcall sub_3109500(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rsi
  int *v9; // r9
  __int64 v10; // rdi
  const char *v11; // rax
  __int64 v12; // rdx
  size_t v13; // rbx
  int v14; // eax
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  char v18; // bl
  __int64 v19; // rax
  __int64 v20; // rcx
  int v22; // eax
  __int64 *v23; // rsi
  const char *v24; // rbx
  size_t v25; // rdx
  size_t v26; // r15
  int v27; // eax
  __int64 *v28; // rsi
  char *v29; // rdx
  unsigned int v30; // edx
  unsigned int v31; // edx
  char v32; // dl
  const void *v33; // [rsp+0h] [rbp-D0h]
  __int64 v34; // [rsp+8h] [rbp-C8h]
  __int64 v35; // [rsp+8h] [rbp-C8h]
  __int64 v36; // [rsp+8h] [rbp-C8h]
  __int64 v37; // [rsp+8h] [rbp-C8h]
  __int64 v38; // [rsp+8h] [rbp-C8h]
  __int64 v39[2]; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v41[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+40h] [rbp-90h] BYREF
  __int64 v43; // [rsp+50h] [rbp-80h] BYREF
  __int64 v44; // [rsp+58h] [rbp-78h]
  char *v45; // [rsp+60h] [rbp-70h]
  unsigned __int64 v46; // [rsp+68h] [rbp-68h] BYREF
  unsigned int v47; // [rsp+70h] [rbp-60h]
  unsigned __int64 v48; // [rsp+78h] [rbp-58h] BYREF
  unsigned int v49; // [rsp+80h] [rbp-50h]
  char v50; // [rsp+88h] [rbp-48h]
  char v51; // [rsp+90h] [rbp-40h]

  v6 = sub_B491C0((__int64)a3);
  v34 = sub_BC1CD0(*(_QWORD *)(a2 + 16), &unk_4F8FAE8, v6) + 8;
  v7 = sub_B43CB0((__int64)a3);
  if ( *(_DWORD *)(a2 + 112) != 1 )
  {
    v24 = sub_BD5D20(v7);
    v26 = v25;
    v27 = sub_C92610();
    if ( (unsigned int)sub_C92860((__int64 *)(a2 + 160), v24, v26, v27) == -1 )
    {
      v28 = *(__int64 **)(a2 + 80);
      if ( v28 )
        sub_30CC6B0((__int64)a1, v28, (__int64)a3, 0);
      else
        *a1 = 0;
      return a1;
    }
  }
  v8 = a3[6];
  v9 = (int *)(a2 + 120);
  v43 = v8;
  if ( v8 )
  {
    sub_B96E90((__int64)&v43, v8, 1);
    v9 = (int *)(a2 + 120);
  }
  sub_30CB1F0(v39, (__int64)&v43, v9);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  v10 = *(a3 - 4);
  if ( v10 )
  {
    if ( *(_BYTE *)v10 )
    {
      v10 = 0;
    }
    else if ( *(_QWORD *)(v10 + 24) != a3[10] )
    {
      v10 = 0;
    }
  }
  v11 = sub_BD5D20(v10);
  v45 = (char *)v39;
  v43 = (__int64)v11;
  v44 = v12;
  LOWORD(v47) = 1029;
  sub_CA0F50(v41, (void **)&v43);
  v13 = v41[1];
  v33 = (const void *)v41[0];
  v14 = sub_C92610();
  v15 = sub_C92860((__int64 *)(a2 + 136), v33, v13, v14);
  if ( v15 == -1 || (v16 = *(_QWORD *)(a2 + 136), v17 = v16 + 8LL * v15, v17 == v16 + 8LL * *(unsigned int *)(a2 + 144)) )
  {
    v22 = *(_DWORD *)(a2 + 116);
    if ( v22 == 1 )
    {
      v44 = 0;
      v43 = 0x80000000LL;
      v45 = "AlwaysInline Fallback";
      v50 = 0;
      v51 = 1;
    }
    else
    {
      if ( v22 != 2 )
      {
        v23 = *(__int64 **)(a2 + 80);
        if ( v23 )
          sub_30CC6B0((__int64)a1, v23, (__int64)a3, 0);
        else
          *a1 = 0;
        goto LABEL_17;
      }
      v51 = 0;
    }
    v18 = *(_BYTE *)(a2 + 128);
    v19 = sub_22077B0(0x98u);
    if ( !v19 )
      goto LABEL_50;
  }
  else
  {
    v18 = *(_BYTE *)(a2 + 128);
    if ( !*(_BYTE *)(*(_QWORD *)v17 + 8LL) )
    {
      v51 = 0;
      v19 = sub_22077B0(0x98u);
      if ( v19 )
        goto LABEL_14;
LABEL_50:
      v32 = v51;
      goto LABEL_51;
    }
    v44 = 0;
    v43 = 0x80000000LL;
    v45 = "previously inlined";
    v50 = 0;
    v51 = 1;
    v19 = sub_22077B0(0x98u);
    if ( !v19 )
      goto LABEL_50;
  }
LABEL_14:
  v20 = v34;
  v35 = v19;
  sub_30CABE0(v19, a2, a3, v20, v51);
  v19 = v35;
  *(_QWORD *)(v35 + 64) = a3;
  *(_QWORD *)v35 = &unk_4A32518;
  *(_BYTE *)(v35 + 136) = 0;
  if ( v51 )
  {
    *(_QWORD *)(v35 + 72) = v43;
    *(_DWORD *)(v35 + 80) = v44;
    v29 = v45;
    *(_BYTE *)(v35 + 128) = 0;
    *(_QWORD *)(v35 + 88) = v29;
    if ( !v50 )
    {
      *(_BYTE *)(v35 + 136) = 1;
      *(_BYTE *)(v35 + 144) = v18;
      goto LABEL_33;
    }
    v30 = v47;
    *(_DWORD *)(v35 + 104) = v47;
    if ( v30 > 0x40 )
    {
      sub_C43780(v35 + 96, (const void **)&v46);
      v19 = v35;
    }
    else
    {
      *(_QWORD *)(v35 + 96) = v46;
    }
    v31 = v49;
    *(_DWORD *)(v19 + 120) = v49;
    if ( v31 > 0x40 )
    {
      v38 = v19;
      sub_C43780(v19 + 112, (const void **)&v48);
      v19 = v38;
    }
    else
    {
      *(_QWORD *)(v19 + 112) = v48;
    }
    *(_BYTE *)(v19 + 128) = 1;
    v32 = v51;
    *(_BYTE *)(v19 + 136) = 1;
    *(_BYTE *)(v19 + 144) = v18;
LABEL_51:
    if ( !v32 )
      goto LABEL_16;
LABEL_33:
    v51 = 0;
    if ( v50 )
    {
      v50 = 0;
      if ( v49 > 0x40 && v48 )
      {
        v36 = v19;
        j_j___libc_free_0_0(v48);
        v19 = v36;
      }
      if ( v47 > 0x40 && v46 )
      {
        v37 = v19;
        j_j___libc_free_0_0(v46);
        v19 = v37;
      }
    }
    goto LABEL_16;
  }
  *(_BYTE *)(v35 + 144) = v18;
LABEL_16:
  *a1 = v19;
LABEL_17:
  if ( (__int64 *)v41[0] != &v42 )
    j_j___libc_free_0(v41[0]);
  if ( (__int64 *)v39[0] != &v40 )
    j_j___libc_free_0(v39[0]);
  return a1;
}
