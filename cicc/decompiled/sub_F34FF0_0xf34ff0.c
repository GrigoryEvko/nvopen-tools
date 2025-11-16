// Function: sub_F34FF0
// Address: 0xf34ff0
//
__int64 __fastcall sub_F34FF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6, void **a7)
{
  char v7; // al
  char v11; // si
  __int64 v12; // rbx
  char i; // dl
  int v14; // ecx
  unsigned int v15; // ecx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r12
  int v19; // ecx
  __int64 v20; // rsi
  int v21; // ecx
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 *v25; // rdi
  __int64 v26; // r15
  __int64 v27; // rdx
  int v28; // ecx
  __int64 v29; // rax
  __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // r14
  char *v34; // rax
  char v35; // dl
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int64 *v38; // rax
  __int64 v39; // r10
  __int64 v40; // rax
  unsigned __int64 *v41; // rax
  unsigned __int64 *v42; // rsi
  __int64 v43; // rax
  unsigned __int64 *v44; // rdi
  __int64 v46; // rdx
  int v47; // eax
  unsigned __int64 v49; // [rsp+20h] [rbp-150h]
  __int64 v51[2]; // [rsp+30h] [rbp-140h] BYREF
  __int64 v52; // [rsp+40h] [rbp-130h] BYREF
  __int64 v53; // [rsp+50h] [rbp-120h] BYREF
  char *v54; // [rsp+58h] [rbp-118h]
  __int64 v55; // [rsp+60h] [rbp-110h]
  int v56; // [rsp+68h] [rbp-108h]
  char v57; // [rsp+6Ch] [rbp-104h]
  char v58; // [rsp+70h] [rbp-100h] BYREF
  unsigned __int64 *v59; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+B8h] [rbp-B8h]
  _QWORD v61[2]; // [rsp+C0h] [rbp-B0h] BYREF
  __int16 v62; // [rsp+D0h] [rbp-A0h]

  v7 = 0;
  v11 = a3;
  v12 = a3;
  for ( i = BYTE1(a3); ; i = 0 )
  {
    if ( !a2 )
      BUG();
    v14 = *((unsigned __int8 *)a2 - 24);
    if ( (_BYTE)v14 != 84 )
    {
      v15 = v14 - 39;
      if ( v15 > 0x38 || ((1LL << v15) & 0x100060000000001LL) == 0 )
        break;
    }
    a2 = (__int64 *)a2[1];
    v7 = 1;
    v11 = 0;
  }
  if ( v7 )
  {
    LOBYTE(v12) = v11;
    BYTE1(v12) = i;
  }
  sub_CA0F50(v51, a7);
  if ( v51[1] )
  {
    v59 = (unsigned __int64 *)v51;
    v62 = 260;
  }
  else
  {
    v59 = (unsigned __int64 *)sub_BD5D20(a1);
    v62 = 773;
    v60 = v46;
    v61[0] = ".split";
  }
  v18 = sub_AA8550((_QWORD *)a1, a2, v12, (__int64)&v59, 1);
  if ( a5 )
  {
    v19 = *(_DWORD *)(a5 + 24);
    v20 = *(_QWORD *)(a5 + 8);
    if ( v19 )
    {
      v21 = v19 - 1;
      v22 = v21 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v23 = (__int64 *)(v20 + 16LL * v22);
      v24 = *v23;
      if ( a1 == *v23 )
      {
LABEL_13:
        v25 = (__int64 *)v23[1];
        if ( v25 )
          sub_D4F330(v25, v18, a5);
      }
      else
      {
        v47 = 1;
        while ( v24 != -4096 )
        {
          v16 = (unsigned int)(v47 + 1);
          v22 = v21 & (v47 + v22);
          v23 = (__int64 *)(v20 + 16LL * v22);
          v24 = *v23;
          if ( a1 == *v23 )
            goto LABEL_13;
          v47 = v16;
        }
      }
    }
  }
  if ( a4 )
  {
    v57 = 1;
    v59 = v61;
    v54 = &v58;
    v55 = 8;
    v56 = 0;
    v61[1] = a1 & 0xFFFFFFFFFFFFFFFBLL;
    v49 = a1 & 0xFFFFFFFFFFFFFFFBLL;
    v61[0] = v18;
    v60 = 0x800000001LL;
    v26 = *(_QWORD *)(v18 + 16);
    v53 = 0;
    if ( v26 )
    {
      v27 = v26;
      while ( 1 )
      {
        v28 = **(unsigned __int8 **)(v27 + 24);
        v29 = v27;
        v27 = *(_QWORD *)(v27 + 8);
        v30 = (unsigned int)(v28 - 30);
        if ( (unsigned __int8)v30 <= 0xAu )
          break;
        if ( !v27 )
        {
          v32 = *(_QWORD *)(v26 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v32 - 30) > 0xAu )
            goto LABEL_51;
          goto LABEL_25;
        }
      }
      v30 = 0;
      while ( 1 )
      {
        v29 = *(_QWORD *)(v29 + 8);
        if ( !v29 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v29 + 24) - 30) <= 0xAu )
        {
          v29 = *(_QWORD *)(v29 + 8);
          ++v30;
          if ( !v29 )
            goto LABEL_23;
        }
      }
LABEL_23:
      v31 = (unsigned int)(2 * v30 + 2) + 1LL;
      if ( v31 <= 8 || (sub_C8D5F0((__int64)&v59, v61, v31, 0x10u, v16, v17), (v26 = *(_QWORD *)(v18 + 16)) != 0) )
      {
        while ( 1 )
        {
          v32 = *(_QWORD *)(v26 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v32 - 30) <= 0xAu )
            break;
LABEL_51:
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            goto LABEL_39;
        }
LABEL_25:
        v33 = *(_QWORD *)(v32 + 40);
        if ( !v57 )
          goto LABEL_33;
LABEL_26:
        v34 = v54;
        v30 = HIDWORD(v55);
        v32 = (__int64)&v54[8 * HIDWORD(v55)];
        if ( v54 != (char *)v32 )
        {
          do
          {
            if ( v33 == *(_QWORD *)v34 )
              goto LABEL_30;
            v34 += 8;
          }
          while ( (char *)v32 != v34 );
        }
        if ( HIDWORD(v55) < (unsigned int)v55 )
        {
          ++HIDWORD(v55);
          *(_QWORD *)v32 = v33;
          ++v53;
LABEL_34:
          v36 = (unsigned int)v60;
          v37 = (unsigned int)v60 + 1LL;
          if ( v37 > HIDWORD(v60) )
          {
            sub_C8D5F0((__int64)&v59, v61, v37, 0x10u, v16, v17);
            v36 = (unsigned int)v60;
          }
          v38 = &v59[2 * v36];
          *v38 = v33;
          v38[1] = v18 & 0xFFFFFFFFFFFFFFFBLL;
          v39 = v49 | 4;
          v30 = HIDWORD(v60);
          LODWORD(v60) = v60 + 1;
          v40 = (unsigned int)v60;
          if ( (unsigned __int64)(unsigned int)v60 + 1 > HIDWORD(v60) )
          {
            sub_C8D5F0((__int64)&v59, v61, (unsigned int)v60 + 1LL, 0x10u, v16, v17);
            v40 = (unsigned int)v60;
            v39 = v49 | 4;
          }
          v41 = &v59[2 * v40];
          *v41 = v33;
          v41[1] = v39;
          LODWORD(v60) = v60 + 1;
          v26 = *(_QWORD *)(v26 + 8);
          if ( v26 )
            goto LABEL_31;
          goto LABEL_39;
        }
LABEL_33:
        sub_C8CC70((__int64)&v53, v33, v32, v30, v16, v17);
        if ( v35 )
          goto LABEL_34;
LABEL_30:
        while ( 1 )
        {
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            break;
LABEL_31:
          v32 = *(_QWORD *)(v26 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v32 - 30) <= 0xAu )
          {
            v33 = *(_QWORD *)(v32 + 40);
            if ( v57 )
              goto LABEL_26;
            goto LABEL_33;
          }
        }
      }
    }
LABEL_39:
    v42 = v59;
    sub_FFB3D0(a4, v59, (unsigned int)v60);
    if ( a6 )
    {
      v43 = sub_FFD350(a4);
      v42 = v59;
      sub_D75690(a6, v59, (unsigned int)v60, v43, 0);
      if ( byte_4F8F8E8[0] )
      {
        v42 = 0;
        nullsub_390(*a6, 0);
      }
    }
    if ( v57 )
    {
      v44 = v59;
      if ( v59 == v61 )
        goto LABEL_45;
      goto LABEL_44;
    }
    _libc_free(v54, v42);
    v44 = v59;
    if ( v59 != v61 )
LABEL_44:
      _libc_free(v44, v42);
  }
LABEL_45:
  if ( (__int64 *)v51[0] != &v52 )
    j_j___libc_free_0(v51[0], v52 + 1);
  return v18;
}
