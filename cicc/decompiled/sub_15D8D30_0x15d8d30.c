// Function: sub_15D8D30
// Address: 0x15d8d30
//
void __fastcall sub_15D8D30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 *v8; // rcx
  char *v9; // r15
  int v10; // eax
  char *v11; // rdx
  __int64 v12; // rax
  char **v13; // r13
  char **v14; // r15
  __int64 v15; // r9
  char *v16; // r10
  __int64 v17; // rax
  char **v18; // rax
  char *v19; // rsi
  unsigned int v20; // edx
  char **v21; // rdi
  char *v22; // r10
  _QWORD *v23; // rbx
  _QWORD *v24; // r13
  unsigned __int64 v25; // rdi
  __int64 *v26; // rbx
  __int64 *v27; // r13
  __int64 v28; // rsi
  __int64 *v29; // rax
  int v30; // edi
  int v31; // r8d
  __int64 v32; // r15
  __int64 *v33; // rax
  int v34; // eax
  __int64 *v35; // rax
  __int64 v36; // rdi
  __int64 *v37; // r13
  char *v38; // [rsp+8h] [rbp-3F8h]
  __int64 v40; // [rsp+58h] [rbp-3A8h]
  __int64 v41; // [rsp+60h] [rbp-3A0h]
  __int64 v42; // [rsp+68h] [rbp-398h]
  __int64 v43; // [rsp+68h] [rbp-398h]
  int v44; // [rsp+74h] [rbp-38Ch]
  __int64 v45[2]; // [rsp+78h] [rbp-388h] BYREF
  __int64 v46; // [rsp+88h] [rbp-378h] BYREF
  char *v47; // [rsp+90h] [rbp-370h] BYREF
  __int64 v48; // [rsp+98h] [rbp-368h] BYREF
  __int64 v49[3]; // [rsp+A0h] [rbp-360h] BYREF
  char v50[8]; // [rsp+B8h] [rbp-348h] BYREF
  _QWORD *v51; // [rsp+C0h] [rbp-340h]
  unsigned int v52; // [rsp+D0h] [rbp-330h]
  __int64 v53; // [rsp+D8h] [rbp-328h]
  char **v54; // [rsp+E0h] [rbp-320h] BYREF
  int v55; // [rsp+E8h] [rbp-318h]
  char v56; // [rsp+F0h] [rbp-310h] BYREF
  __int64 *v57; // [rsp+130h] [rbp-2D0h] BYREF
  __int64 v58; // [rsp+138h] [rbp-2C8h]
  _BYTE v59[128]; // [rsp+140h] [rbp-2C0h] BYREF
  char *v60; // [rsp+1C0h] [rbp-240h] BYREF
  __int64 v61; // [rsp+1C8h] [rbp-238h]
  _QWORD v62[70]; // [rsp+1D0h] [rbp-230h] BYREF

  v4 = a2;
  v45[0] = a3;
  v40 = sub_15CC960(a1, a3);
  if ( !v40 )
  {
    v32 = sub_15CC960(a1, 0);
    v33 = (__int64 *)sub_22077B0(56);
    v40 = (__int64)v33;
    if ( v33 )
    {
      *v33 = a3;
      v33[1] = v32;
      v34 = 0;
      if ( v32 )
        v34 = *(_DWORD *)(v32 + 16) + 1;
      *(_DWORD *)(v40 + 16) = v34;
      *(_QWORD *)(v40 + 24) = 0;
      *(_QWORD *)(v40 + 32) = 0;
      *(_QWORD *)(v40 + 40) = 0;
      *(_QWORD *)(v40 + 48) = -1;
    }
    v60 = (char *)v40;
    sub_15CE4A0(v32 + 24, &v60);
    v35 = sub_15CFF10(a1 + 48, v45);
    v36 = v35[1];
    v37 = v35;
    v35[1] = v40;
    if ( v36 )
    {
      sub_15CBC60(v36);
      v40 = v37[1];
    }
    sub_15CDD90(a1, v45);
  }
  *(_BYTE *)(a1 + 96) = 0;
  v8 = (__int64 *)sub_15CC960(a1, a4);
  if ( v8 )
  {
    sub_15D8000(a1, a2, (__int64 *)v40, v8);
    return;
  }
  v9 = v50;
  v57 = (__int64 *)v59;
  v58 = 0x800000000LL;
  sub_15CDF00((__int64)v49, a2);
  v60 = (char *)v62;
  v61 = 0x4000000001LL;
  v46 = a4;
  v62[0] = a4;
  v48 = a4;
  if ( (unsigned __int8)sub_15CE6E0((__int64)v50, &v48, &v54) )
    *(_DWORD *)(sub_15D4720((__int64)v50, &v46) + 12) = 0;
  v10 = v61;
  v44 = 0;
  if ( (_DWORD)v61 )
  {
LABEL_10:
    v11 = *(char **)&v60[8 * v10 - 8];
    LODWORD(v61) = v10 - 1;
    v47 = v11;
    v12 = sub_15D4720((__int64)v9, (__int64 *)&v47);
    if ( *(_DWORD *)(v12 + 8) )
      goto LABEL_9;
    ++v44;
    *(_QWORD *)(v12 + 24) = v47;
    *(_DWORD *)(v12 + 16) = v44;
    *(_DWORD *)(v12 + 8) = v44;
    sub_15CE600((__int64)v49, &v47);
    sub_15CF0D0((__int64)&v54, (__int64)v47, v53);
    v13 = &v54[v55];
    if ( v54 == v13 )
      goto LABEL_25;
    v41 = (__int64)v9;
    v14 = v54;
    while ( 1 )
    {
      v19 = *v14;
      v48 = (__int64)*v14;
      if ( !v52 )
        goto LABEL_13;
      v20 = (v52 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v21 = (char **)&v51[9 * v20];
      v22 = *v21;
      if ( v19 != *v21 )
        break;
LABEL_20:
      if ( v21 == &v51[9 * v52] || !*((_DWORD *)v21 + 2) )
        goto LABEL_13;
      if ( v19 == v47 )
      {
LABEL_17:
        if ( v13 == ++v14 )
          goto LABEL_24;
      }
      else
      {
        ++v14;
        sub_15CDD90((__int64)(v21 + 5), &v47);
        if ( v13 == v14 )
        {
LABEL_24:
          v9 = (char *)v41;
          v13 = v54;
LABEL_25:
          if ( v13 != (char **)&v56 )
          {
            _libc_free((unsigned __int64)v13);
            v10 = v61;
            if ( !(_DWORD)v61 )
            {
LABEL_27:
              v4 = a2;
              goto LABEL_28;
            }
            goto LABEL_10;
          }
LABEL_9:
          v10 = v61;
          if ( !(_DWORD)v61 )
            goto LABEL_27;
          goto LABEL_10;
        }
      }
    }
    v30 = 1;
    while ( v22 != (char *)-8LL )
    {
      v31 = v30 + 1;
      v20 = (v52 - 1) & (v30 + v20);
      v21 = (char **)&v51[9 * v20];
      v22 = *v21;
      if ( v19 == *v21 )
        goto LABEL_20;
      v30 = v31;
    }
LABEL_13:
    v15 = sub_15CC960(a1, (__int64)v19);
    if ( v15 )
    {
      v16 = v47;
      v17 = (unsigned int)v58;
      if ( (unsigned int)v58 >= HIDWORD(v58) )
      {
        v38 = v47;
        v43 = v15;
        sub_16CD150(&v57, v59, 0, 16);
        v17 = (unsigned int)v58;
        v16 = v38;
        v15 = v43;
      }
      v18 = (char **)&v57[2 * v17];
      *v18 = v16;
      v18[1] = (char *)v15;
      LODWORD(v58) = v58 + 1;
    }
    else
    {
      v42 = sub_15D4720(v41, &v48);
      sub_15CDD90((__int64)&v60, &v48);
      *(_DWORD *)(v42 + 12) = v44;
      sub_15CDD90(v42 + 40, &v47);
    }
    goto LABEL_17;
  }
LABEL_28:
  if ( v60 != (char *)v62 )
    _libc_free((unsigned __int64)v60);
  sub_15D4AE0(v49, a1, 0);
  sub_15D4EE0((__int64)v49, a1, (_QWORD *)v40);
  if ( v52 )
  {
    v23 = v51;
    v24 = &v51[9 * v52];
    do
    {
      if ( *v23 != -16 && *v23 != -8 )
      {
        v25 = v23[5];
        if ( (_QWORD *)v25 != v23 + 7 )
          _libc_free(v25);
      }
      v23 += 9;
    }
    while ( v24 != v23 );
  }
  j___libc_free_0(v51);
  sub_15CE080(v49);
  v26 = v57;
  v27 = &v57[2 * (unsigned int)v58];
  if ( v57 != v27 )
  {
    do
    {
      v28 = *v26;
      v26 += 2;
      v29 = (__int64 *)sub_15CC960(a1, v28);
      sub_15D8000(a1, v4, v29, (__int64 *)*(v26 - 1));
    }
    while ( v27 != v26 );
    v27 = v57;
  }
  if ( v27 != (__int64 *)v59 )
    _libc_free((unsigned __int64)v27);
}
