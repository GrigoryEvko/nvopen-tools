// Function: sub_3270B90
// Address: 0x3270b90
//
__int64 __fastcall sub_3270B90(int a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  unsigned int v7; // r15d
  int v8; // eax
  __int64 v10; // rcx
  __int64 *v11; // rax
  __int64 v12; // r9
  unsigned __int16 *v13; // r8
  int v14; // r15d
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // edx
  unsigned int v19; // eax
  __int128 v20; // rax
  int v21; // r9d
  int v22; // esi
  __int64 *v23; // rax
  __int64 v24; // r10
  __int64 *v25; // rax
  __int64 v26; // rsi
  char v27; // al
  char v28; // al
  unsigned int v29; // eax
  unsigned int v30; // eax
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-D8h]
  __int128 v34; // [rsp+10h] [rbp-D0h]
  __int128 v35; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+10h] [rbp-D0h]
  __int64 v37; // [rsp+10h] [rbp-D0h]
  __int128 v38; // [rsp+10h] [rbp-D0h]
  __int128 v39; // [rsp+10h] [rbp-D0h]
  __int64 v40; // [rsp+10h] [rbp-D0h]
  __int64 v41; // [rsp+20h] [rbp-C0h]
  __int128 v42; // [rsp+20h] [rbp-C0h]
  __int64 v43; // [rsp+20h] [rbp-C0h]
  __int64 v44; // [rsp+20h] [rbp-C0h]
  __int64 v45; // [rsp+20h] [rbp-C0h]
  __int64 v46; // [rsp+30h] [rbp-B0h]
  __int64 v47; // [rsp+30h] [rbp-B0h]
  __int64 v48; // [rsp+38h] [rbp-A8h]
  char v49; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v50; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v51; // [rsp+48h] [rbp-98h]
  unsigned __int64 v52; // [rsp+50h] [rbp-90h] BYREF
  __int64 v53; // [rsp+58h] [rbp-88h]
  __int64 v54; // [rsp+60h] [rbp-80h]
  __int64 v55; // [rsp+68h] [rbp-78h]
  unsigned __int64 v56; // [rsp+70h] [rbp-70h] BYREF
  __int64 v57; // [rsp+78h] [rbp-68h]
  __int64 v58; // [rsp+80h] [rbp-60h]
  unsigned __int64 v59; // [rsp+88h] [rbp-58h] BYREF
  __int64 v60; // [rsp+90h] [rbp-50h]
  __int64 v61; // [rsp+98h] [rbp-48h]
  _DWORD *v62; // [rsp+A0h] [rbp-40h]

  if ( a1 == 56 )
  {
    v6 = a2[5];
    v7 = *((_DWORD *)a2 + 12);
  }
  else
  {
    v6 = *a2;
    v7 = *((_DWORD *)a2 + 2);
    a2 += 5;
  }
  v8 = *(_DWORD *)(v6 + 24);
  if ( v8 != 35 && v8 != 11 )
    return 0;
  v10 = *a2;
  if ( *(_DWORD *)(*a2 + 24) != 214 )
    return 0;
  if ( *(_WORD *)(*(_QWORD *)(**(_QWORD **)(v10 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v10 + 40) + 8LL)) != 2 )
    return 0;
  v54 = 0;
  v52 = 208;
  v53 = 186;
  v55 = 0;
  v56 = 1;
  v57 = 64;
  v58 = 0;
  v59 = 0;
  v60 = 64;
  v61 = 0x100000011LL;
  v62 = 0;
  v11 = *(__int64 **)(v10 + 40);
  v12 = *v11;
  if ( *(_DWORD *)(*v11 + 24) != 208 )
    return 0;
  v23 = *(__int64 **)(v12 + 40);
  v24 = *v23;
  if ( *(_DWORD *)(*v23 + 24) != 186 )
    return 0;
  v25 = *(__int64 **)(v24 + 40);
  if ( *v25 )
  {
    v33 = v12;
    v47 = v24;
    v40 = v10;
    v31 = sub_32657E0((__int64)&v56, v25[5]);
    v10 = v40;
    v24 = v47;
    v12 = v33;
    if ( v31 )
      goto LABEL_28;
    v25 = *(__int64 **)(v47 + 40);
    v26 = v25[5];
  }
  else
  {
    v26 = v25[5];
  }
  if ( !v26 )
    goto LABEL_31;
  v46 = v12;
  v36 = v24;
  v43 = v10;
  v27 = sub_32657E0((__int64)&v56, *v25);
  v10 = v43;
  v24 = v36;
  v49 = v27;
  v12 = v46;
  if ( !v27 )
  {
LABEL_50:
    v29 = v60;
    goto LABEL_32;
  }
LABEL_28:
  if ( !BYTE4(v58) || (_DWORD)v58 == ((unsigned int)v58 & *(_DWORD *)(v24 + 28)) )
  {
    v37 = v10;
    v44 = v12;
    v28 = sub_32657E0((__int64)&v59, *(_QWORD *)(*(_QWORD *)(v12 + 40) + 40LL));
    v10 = v37;
    v49 = v28;
    if ( v28 )
    {
      v32 = *(_QWORD *)(*(_QWORD *)(v44 + 40) + 80LL);
      if ( *(_DWORD *)(v32 + 24) == 8 && (!BYTE4(v61) || (_DWORD)v61 == *(_DWORD *)(v32 + 96)) )
      {
        if ( v62 )
          *v62 = *(_DWORD *)(v32 + 96);
        goto LABEL_50;
      }
    }
  }
LABEL_31:
  v49 = 0;
  v29 = v60;
LABEL_32:
  if ( v29 > 0x40 && v59 )
  {
    v45 = v10;
    j_j___libc_free_0_0(v59);
    v10 = v45;
  }
  if ( (unsigned int)v57 > 0x40 && v56 )
  {
    v41 = v10;
    j_j___libc_free_0_0(v56);
    v10 = v41;
  }
  if ( !v49 )
    return 0;
  v13 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * v7);
  v14 = *v13;
  v15 = *(_QWORD **)(**(_QWORD **)(v10 + 40) + 40LL);
  v48 = *((_QWORD *)v13 + 1);
  *(_QWORD *)&v42 = sub_33FB310(a4, *v15, v15[1], a3, *v13, v48);
  v16 = *(_QWORD *)(v6 + 96);
  *((_QWORD *)&v42 + 1) = v17;
  v18 = *(_DWORD *)(v16 + 32);
  v51 = v18;
  if ( a1 == 56 )
  {
    if ( v18 > 0x40 )
      sub_C43780((__int64)&v50, (const void **)(v16 + 24));
    else
      v50 = *(_QWORD *)(v16 + 24);
    sub_C46A40((__int64)&v50, 1);
    v30 = v51;
    v51 = 0;
    LODWORD(v53) = v30;
    v52 = v50;
    *(_QWORD *)&v20 = sub_34007B0(a4, (unsigned int)&v52, a3, v14, v48, 0, 0);
    if ( (unsigned int)v53 > 0x40 && v52 )
    {
      v38 = v20;
      j_j___libc_free_0_0(v52);
      v20 = v38;
    }
    if ( v51 > 0x40 && v50 )
    {
      v39 = v20;
      j_j___libc_free_0_0(v50);
      v20 = v39;
    }
    v22 = 57;
  }
  else
  {
    if ( v18 > 0x40 )
      sub_C43780((__int64)&v50, (const void **)(v16 + 24));
    else
      v50 = *(_QWORD *)(v16 + 24);
    sub_C46F20((__int64)&v50, 1u);
    v19 = v51;
    v51 = 0;
    LODWORD(v53) = v19;
    v52 = v50;
    *(_QWORD *)&v20 = sub_34007B0(a4, (unsigned int)&v52, a3, v14, v48, 0, 0);
    if ( (unsigned int)v53 > 0x40 && v52 )
    {
      v34 = v20;
      j_j___libc_free_0_0(v52);
      v20 = v34;
    }
    if ( v51 > 0x40 )
    {
      if ( v50 )
      {
        v35 = v20;
        j_j___libc_free_0_0(v50);
        v20 = v35;
      }
    }
    v22 = 56;
  }
  return sub_3406EB0(a4, v22, a3, v14, v48, v21, v20, v42);
}
