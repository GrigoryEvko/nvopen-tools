// Function: sub_2F3B580
// Address: 0x2f3b580
//
__int64 __fastcall sub_2F3B580(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r15
  __int64 (*v11)(void); // rdx
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rbx
  unsigned __int64 v15; // rdi
  __int64 **v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  void **v21; // rax
  __int64 **v22; // rdx
  __int64 v23; // rcx
  __int64 **v24; // rax
  int v25; // eax
  __int64 **v26; // rax
  __int64 **v27; // rdx
  __int64 v28; // rcx
  __int64 **v29; // rax
  int v30; // eax
  __int64 **v31; // rax
  __int64 **v32; // rdi
  void **v33; // rdi
  __int64 *v34; // rax
  __int64 *v35; // rax
  __int64 **v36; // rdi
  __int64 v37; // [rsp+0h] [rbp-200h]
  __int64 v38; // [rsp+8h] [rbp-1F8h]
  __int64 v39; // [rsp+10h] [rbp-1F0h] BYREF
  void **v40; // [rsp+18h] [rbp-1E8h]
  unsigned int v41; // [rsp+20h] [rbp-1E0h]
  unsigned int v42; // [rsp+24h] [rbp-1DCh]
  char v43; // [rsp+2Ch] [rbp-1D4h]
  char v44[16]; // [rsp+30h] [rbp-1D0h] BYREF
  __int64 v45; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 **v46; // [rsp+48h] [rbp-1B8h]
  unsigned int v47; // [rsp+54h] [rbp-1ACh]
  int v48; // [rsp+58h] [rbp-1A8h]
  char v49; // [rsp+5Ch] [rbp-1A4h]
  char v50[16]; // [rsp+60h] [rbp-1A0h] BYREF
  _QWORD v51[4]; // [rsp+70h] [rbp-190h] BYREF
  __int64 v52[4]; // [rsp+90h] [rbp-170h] BYREF
  char *v53; // [rsp+B0h] [rbp-150h]
  char v54; // [rsp+C8h] [rbp-138h] BYREF
  char *v55; // [rsp+E8h] [rbp-118h]
  char v56; // [rsp+100h] [rbp-100h] BYREF
  char *v57; // [rsp+128h] [rbp-D8h]
  char v58; // [rsp+138h] [rbp-C8h] BYREF
  char *v59; // [rsp+170h] [rbp-90h]
  char v60; // [rsp+180h] [rbp-80h] BYREF
  unsigned __int64 v61; // [rsp+1B8h] [rbp-48h]

  v38 = sub_2EB2140(a4, &qword_50208B0, a3) + 8;
  v7 = sub_2EB2140(a4, &qword_50209D0, a3);
  v8 = sub_BC1CD0(*(_QWORD *)(v7 + 8), &unk_4F86540, *(_QWORD *)a3);
  v9 = *a2;
  v10 = v8 + 8;
  v11 = *(__int64 (**)(void))(**(_QWORD **)(a3 + 16) + 128LL);
  v12 = 0;
  if ( v11 != sub_2DAC790 )
  {
    v37 = *a2;
    v12 = v11();
    v9 = v37;
  }
  v51[0] = v12;
  v51[2] = v10;
  v51[1] = v38;
  v51[3] = v9;
  sub_2F5FEE0(v52);
  if ( !(unsigned __int8)sub_2F3AD20(v51, a3) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_5;
  }
  sub_2EAFFB0((__int64)&v39);
  if ( v47 != v48 )
    goto LABEL_23;
  if ( v43 )
  {
    v21 = v40;
    v36 = (__int64 **)&v40[v42];
    v18 = v42;
    v17 = (__int64 **)v40;
    if ( v40 == (void **)v36 )
      goto LABEL_81;
    while ( *v17 != &qword_4F82400 )
    {
      if ( v36 == ++v17 )
      {
LABEL_27:
        while ( *v21 != &unk_4F82408 )
        {
          if ( ++v21 == (void **)v17 )
            goto LABEL_81;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v39, (__int64)&qword_4F82400) )
  {
LABEL_23:
    if ( !v43 )
    {
LABEL_83:
      sub_C8CC70((__int64)&v39, (__int64)&unk_4F82408, (__int64)v17, v18, v19, v20);
      goto LABEL_28;
    }
    v21 = v40;
    v18 = v42;
    v17 = (__int64 **)&v40[v42];
    if ( v40 != (void **)v17 )
      goto LABEL_27;
LABEL_81:
    if ( v41 > (unsigned int)v18 )
    {
      v42 = v18 + 1;
      *v17 = (__int64 *)&unk_4F82408;
      ++v39;
      goto LABEL_28;
    }
    goto LABEL_83;
  }
LABEL_28:
  if ( v49 )
  {
    v22 = &v46[v47];
    v23 = v47;
    if ( v46 == v22 )
    {
LABEL_65:
      if ( v48 != v47 )
        goto LABEL_35;
      goto LABEL_66;
    }
    v24 = v46;
    while ( *v24 != qword_501FE48 )
    {
      if ( v22 == ++v24 )
        goto LABEL_65;
    }
    v22 = (__int64 **)v46[--v47];
    *v24 = (__int64 *)v22;
    v23 = v47;
    ++v45;
    v25 = v48;
  }
  else
  {
    v35 = sub_C8CA60((__int64)&v45, (__int64)qword_501FE48);
    if ( v35 )
    {
      *v35 = -2;
      ++v45;
      v23 = v47;
      v25 = ++v48;
    }
    else
    {
      v23 = v47;
      v25 = v48;
    }
  }
  if ( v25 != (_DWORD)v23 )
  {
LABEL_35:
    if ( !v43 )
    {
LABEL_80:
      sub_C8CC70((__int64)&v39, (__int64)qword_501FE48, (__int64)v22, v23, v19, v20);
      goto LABEL_40;
    }
    v26 = (__int64 **)v40;
    v23 = v42;
    v22 = (__int64 **)&v40[v42];
    if ( v22 != (__int64 **)v40 )
      goto LABEL_39;
LABEL_78:
    if ( (unsigned int)v23 < v41 )
    {
      v42 = v23 + 1;
      *v22 = qword_501FE48;
      ++v39;
      goto LABEL_40;
    }
    goto LABEL_80;
  }
LABEL_66:
  if ( v43 )
  {
    v26 = (__int64 **)v40;
    v33 = &v40[v42];
    v23 = v42;
    v22 = (__int64 **)v40;
    if ( v40 == v33 )
      goto LABEL_78;
    while ( *v22 != &qword_4F82400 )
    {
      if ( v33 == (void **)++v22 )
      {
LABEL_39:
        while ( *v26 != qword_501FE48 )
        {
          if ( ++v26 == v22 )
            goto LABEL_78;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v39, (__int64)&qword_4F82400) )
  {
    goto LABEL_35;
  }
LABEL_40:
  if ( !v49 )
  {
    v34 = sub_C8CA60((__int64)&v45, (__int64)&qword_50208B0);
    if ( !v34 )
    {
      v28 = v47;
      goto LABEL_58;
    }
    *v34 = -2;
    ++v45;
    v28 = v47;
    v30 = ++v48;
LABEL_46:
    if ( (_DWORD)v28 != v30 )
      goto LABEL_47;
    goto LABEL_59;
  }
  v27 = &v46[v47];
  v28 = v47;
  v29 = v46;
  if ( v46 != v27 )
  {
    while ( *v29 != &qword_50208B0 )
    {
      if ( v27 == ++v29 )
        goto LABEL_58;
    }
    v27 = (__int64 **)v46[--v47];
    *v29 = (__int64 *)v27;
    v28 = v47;
    ++v45;
    v30 = v48;
    goto LABEL_46;
  }
LABEL_58:
  if ( (_DWORD)v28 != v48 )
    goto LABEL_47;
LABEL_59:
  if ( v43 )
  {
    v31 = (__int64 **)v40;
    v32 = (__int64 **)&v40[v42];
    v28 = v42;
    v27 = (__int64 **)v40;
    if ( v40 != (void **)v32 )
    {
      while ( *v27 != &qword_4F82400 )
      {
        if ( v32 == ++v27 )
        {
LABEL_51:
          while ( *v31 != &qword_50208B0 )
          {
            if ( v27 == ++v31 )
              goto LABEL_76;
          }
          goto LABEL_52;
        }
      }
      goto LABEL_52;
    }
    goto LABEL_76;
  }
  if ( sub_C8CA60((__int64)&v39, (__int64)&qword_4F82400) )
    goto LABEL_52;
LABEL_47:
  if ( !v43 )
    goto LABEL_56;
  v31 = (__int64 **)v40;
  v28 = v42;
  v27 = (__int64 **)&v40[v42];
  if ( v27 != (__int64 **)v40 )
    goto LABEL_51;
LABEL_76:
  if ( (unsigned int)v28 < v41 )
  {
    v42 = v28 + 1;
    *v27 = &qword_50208B0;
    ++v39;
    goto LABEL_52;
  }
LABEL_56:
  sub_C8CC70((__int64)&v39, (__int64)&qword_50208B0, (__int64)v27, v28, v19, v20);
LABEL_52:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v44, (__int64)&v39);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v50, (__int64)&v45);
  if ( !v49 )
    _libc_free((unsigned __int64)v46);
  if ( !v43 )
    _libc_free((unsigned __int64)v40);
LABEL_5:
  if ( v61 )
    j_j___libc_free_0_0(v61);
  if ( v59 != &v60 )
    _libc_free((unsigned __int64)v59);
  if ( v57 != &v58 )
    _libc_free((unsigned __int64)v57);
  if ( v55 != &v56 )
    _libc_free((unsigned __int64)v55);
  if ( v53 != &v54 )
    _libc_free((unsigned __int64)v53);
  v13 = v52[0];
  if ( v52[0] )
  {
    v14 = v52[0] + 24LL * *(_QWORD *)(v52[0] - 8);
    if ( v52[0] != v14 )
    {
      do
      {
        v15 = *(_QWORD *)(v14 - 8);
        v14 -= 24;
        if ( v15 )
          j_j___libc_free_0_0(v15);
      }
      while ( v13 != v14 );
    }
    j_j_j___libc_free_0_0(v13 - 8);
  }
  return a1;
}
