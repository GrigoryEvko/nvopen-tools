// Function: sub_2D1AB30
// Address: 0x2d1ab30
//
__int64 __fastcall sub_2D1AB30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // si
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r12
  char *v13; // rax
  unsigned __int64 v14; // r8
  _QWORD *v15; // r15
  __int64 v16; // rbx
  int v17; // edx
  __int64 v18; // r14
  _BYTE *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  _BYTE *v24; // r10
  bool v25; // al
  __int64 v26; // rcx
  __int64 v27; // r9
  __int64 v28; // r8
  _BYTE *v29; // r10
  __int64 v30; // rbx
  char *v31; // rax
  __int64 v32; // rdx
  bool v33; // al
  _QWORD *v34; // rax
  __int64 v35; // r12
  __int64 v36; // rcx
  int v37; // r12d
  unsigned int v38; // r15d
  __int64 v39; // rax
  __int64 v40; // r14
  char *v41; // rax
  char *v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // [rsp+8h] [rbp-998h]
  __int64 v51; // [rsp+8h] [rbp-998h]
  __int64 v52; // [rsp+8h] [rbp-998h]
  unsigned __int8 v53; // [rsp+10h] [rbp-990h]
  _BYTE *v54; // [rsp+10h] [rbp-990h]
  __int64 v55; // [rsp+10h] [rbp-990h]
  _BYTE *v56; // [rsp+10h] [rbp-990h]
  _BYTE v57[16]; // [rsp+20h] [rbp-980h] BYREF
  void (__fastcall *v58)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-970h]
  __int64 v59; // [rsp+40h] [rbp-960h] BYREF
  char *v60; // [rsp+48h] [rbp-958h]
  __int64 v61; // [rsp+50h] [rbp-950h]
  int v62; // [rsp+58h] [rbp-948h]
  char v63; // [rsp+5Ch] [rbp-944h]
  char v64; // [rsp+60h] [rbp-940h] BYREF
  _QWORD *v65; // [rsp+160h] [rbp-840h] BYREF
  unsigned int v66; // [rsp+168h] [rbp-838h]
  unsigned int v67; // [rsp+16Ch] [rbp-834h]
  _QWORD v68[262]; // [rsp+170h] [rbp-830h] BYREF

  v6 = 1;
  v7 = (__int64)v68;
  v8 = *(_QWORD *)(a1 + 80);
  v60 = &v64;
  v9 = v8 - 24;
  v59 = 0;
  if ( !v8 )
    v9 = 0;
  v63 = 1;
  v61 = 32;
  v62 = 0;
  v65 = v68;
  v67 = 256;
  v66 = 1;
  v53 = 0;
  v68[0] = v9;
  v10 = 1;
  while ( 1 )
  {
    v11 = v10;
    v12 = *(_QWORD *)(v7 + 8LL * v10 - 8);
    v66 = v10 - 1;
    if ( !v6 )
      goto LABEL_45;
    v13 = v60;
    v7 = HIDWORD(v61);
    v11 = (__int64)&v60[8 * HIDWORD(v61)];
    if ( v60 == (char *)v11 )
    {
LABEL_46:
      if ( HIDWORD(v61) < (unsigned int)v61 )
      {
        ++HIDWORD(v61);
        *(_QWORD *)v11 = v12;
        ++v59;
        goto LABEL_9;
      }
LABEL_45:
      sub_C8CC70((__int64)&v59, v12, v11, v7, a5, a6);
      goto LABEL_9;
    }
    while ( v12 != *(_QWORD *)v13 )
    {
      v13 += 8;
      if ( (char *)v11 == v13 )
        goto LABEL_46;
    }
LABEL_9:
    v14 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v14 == v12 + 48 )
      goto LABEL_30;
    if ( !v14 )
      BUG();
    v15 = (_QWORD *)(v14 - 24);
    v16 = v14 - 24;
    v17 = *(unsigned __int8 *)(v14 - 24);
    if ( (unsigned int)(v17 - 30) > 0xA )
    {
LABEL_30:
      v16 = 0;
LABEL_31:
      v37 = sub_B46E30(v16);
      if ( !v37 )
        goto LABEL_28;
      v38 = 0;
      while ( 1 )
      {
        v39 = sub_B46EC0(v16, v38);
        v40 = v39;
        if ( v63 )
        {
          v41 = v60;
          v42 = &v60[8 * HIDWORD(v61)];
          if ( v60 == v42 )
            goto LABEL_41;
          while ( v40 != *(_QWORD *)v41 )
          {
            v41 += 8;
            if ( v42 == v41 )
              goto LABEL_41;
          }
LABEL_38:
          if ( v37 == ++v38 )
            goto LABEL_28;
        }
        else
        {
          if ( sub_C8CA60((__int64)&v59, v39) )
            goto LABEL_38;
LABEL_41:
          v43 = v66;
          v36 = v67;
          v44 = v66 + 1LL;
          if ( v44 > v67 )
          {
            sub_C8D5F0((__int64)&v65, v68, v44, 8u, a5, a6);
            v43 = v66;
          }
          ++v38;
          v65[v43] = v40;
          ++v66;
          if ( v37 == v38 )
            goto LABEL_28;
        }
      }
    }
    if ( (_BYTE)v17 != 31 || (*(_DWORD *)(v14 - 20) & 0x7FFFFFF) != 3 )
      goto LABEL_31;
    v18 = *(_QWORD *)(v14 - 120);
    v50 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)v18 <= 0x1Cu )
    {
      if ( *(_BYTE *)v18 != 5 )
        goto LABEL_31;
      v19 = (_BYTE *)sub_B2BEC0(a1);
      v46 = sub_97B670((_BYTE *)v18, (__int64)v19, 0);
      v23 = v50;
      v24 = (_BYTE *)v46;
    }
    else
    {
      v19 = (_BYTE *)sub_AA4E30(*(_QWORD *)(v18 + 40));
      v20 = sub_97D880(v18, v19, 0);
      v23 = v50;
      v24 = (_BYTE *)v20;
    }
    v51 = v23;
    if ( !v24 || *v24 != 17 )
      goto LABEL_31;
    v54 = v24;
    v25 = sub_AD7890((__int64)v24, (__int64)v19, v21, v22, v23);
    v28 = v51;
    v29 = v54;
    v30 = *(_QWORD *)(v51 - 32LL * v25 - 56);
    if ( v63 )
    {
      v31 = v60;
      v32 = (__int64)&v60[8 * HIDWORD(v61)];
      if ( v60 != (char *)v32 )
      {
        while ( v30 != *(_QWORD *)v31 )
        {
          v31 += 8;
          if ( (char *)v32 == v31 )
            goto LABEL_59;
        }
        goto LABEL_23;
      }
LABEL_59:
      v48 = v66;
      v26 = v67;
      v49 = v66 + 1LL;
      if ( v49 > v67 )
      {
        v19 = v68;
        v52 = v28;
        v56 = v29;
        sub_C8D5F0((__int64)&v65, v68, v49, 8u, v28, v27);
        v48 = v66;
        v28 = v52;
        v29 = v56;
      }
      v32 = (__int64)v65;
      v65[v48] = v30;
      ++v66;
      goto LABEL_23;
    }
    v19 = *(_BYTE **)(v51 - 32LL * v25 - 56);
    v47 = sub_C8CA60((__int64)&v59, v30);
    v29 = v54;
    v28 = v51;
    if ( !v47 )
      goto LABEL_59;
LABEL_23:
    v55 = v28;
    v33 = sub_AD7A80(v29, (__int64)v19, v32, v26, v28);
    sub_AA5980(*(_QWORD *)(v55 - 32LL * v33 - 56), v12, 0);
    v34 = sub_BD2C40(72, 1u);
    v35 = (__int64)v34;
    if ( v34 )
      sub_B4C8F0((__int64)v34, v30, 1u, v55, 0);
    sub_BD84D0((__int64)v15, v35);
    sub_B43D60(v15);
    v58 = 0;
    sub_F5CAB0((char *)v18, 0, 0, (__int64)v57);
    if ( v58 )
      v58(v57, v57, 3);
    v53 = 1;
LABEL_28:
    v10 = v66;
    if ( !v66 )
      break;
    v7 = (__int64)v65;
    v6 = v63;
  }
  if ( v53 )
    sub_F62E00(a1, 0, 0, v36, a5, a6);
  if ( v65 != v68 )
    _libc_free((unsigned __int64)v65);
  if ( !v63 )
    _libc_free((unsigned __int64)v60);
  return v53;
}
