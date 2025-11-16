// Function: sub_2EBDB80
// Address: 0x2ebdb80
//
__int64 __fastcall sub_2EBDB80(__int64 a1, int a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  unsigned int v5; // r12d
  _BYTE *v6; // rbx
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  void *v29; // rax
  __int64 *v30; // rdi
  void *v31; // rax
  void *v32; // rax
  __int64 *v33; // rbx
  void *v34; // rax
  __int64 v35; // r15
  void *v36; // rax
  __int64 *v37; // rbx
  void *v38; // rax
  __int64 v39; // r15
  void *v40; // rax
  __int64 *v41; // rdi
  __int64 v42; // [rsp+0h] [rbp-10A0h]
  __int64 *v43; // [rsp+0h] [rbp-10A0h]
  __int64 v44; // [rsp+8h] [rbp-1098h]
  __int64 v45; // [rsp+8h] [rbp-1098h]
  __int64 *v46; // [rsp+10h] [rbp-1090h] BYREF
  int v47; // [rsp+18h] [rbp-1088h]
  _BYTE v48[32]; // [rsp+20h] [rbp-1080h] BYREF
  unsigned __int64 v49[2]; // [rsp+40h] [rbp-1060h] BYREF
  _QWORD v50[64]; // [rsp+50h] [rbp-1050h] BYREF
  _BYTE *v51; // [rsp+250h] [rbp-E50h]
  __int64 v52; // [rsp+258h] [rbp-E48h]
  _BYTE v53[3584]; // [rsp+260h] [rbp-E40h] BYREF
  __int64 v54; // [rsp+1060h] [rbp-40h]

  v49[1] = 0x4000000001LL;
  v49[0] = (unsigned __int64)v50;
  v50[0] = 0;
  v51 = v53;
  v52 = 0x4000000000LL;
  v54 = 0;
  if ( !sub_2EBA560(a1) )
    goto LABEL_2;
  if ( !*(_QWORD *)(a1 + 128) && *(_DWORD *)(a1 + 8) )
  {
    v29 = sub_CB72A0();
    sub_904010((__int64)v29, "Tree has no parent but has roots!\n");
    v30 = (__int64 *)sub_CB72A0();
    if ( v30[4] != v30[2] )
      sub_CB5AE0(v30);
    goto LABEL_2;
  }
  sub_2EB9A60(&v46, a1, 0, v2, v3, v4);
  v5 = sub_2EB4750(a1, (__int64)&v46, v10, v11, v12, v13);
  if ( (_BYTE)v5 )
  {
    if ( v46 != (__int64 *)v48 )
      _libc_free((unsigned __int64)v46);
    if ( !(unsigned __int8)sub_2EB8C90((__int64)v49, a1, v14, v15, v16, v17) )
      goto LABEL_2;
    if ( !(unsigned __int8)sub_2EB4410(a1) )
      goto LABEL_2;
    v5 = sub_2EB6B70(a1, a1, v18, v19, v20);
    if ( !(_BYTE)v5 )
      goto LABEL_2;
    if ( (unsigned int)(a2 - 1) <= 1 )
    {
      if ( !(unsigned __int8)sub_2EBCA80((__int64)v49, a1, v21, v22, v23, v24) )
      {
LABEL_2:
        v5 = 0;
        goto LABEL_3;
      }
      if ( a2 == 2 )
        v5 = sub_2EBD8D0((__int64)v49, a1, v25, v26, v27, v28);
    }
  }
  else
  {
    v31 = sub_CB72A0();
    sub_904010((__int64)v31, "Tree has different roots than freshly computed ones!\n");
    v32 = sub_CB72A0();
    sub_904010((__int64)v32, "\tPDT roots: ");
    v33 = *(__int64 **)a1;
    v42 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v42 )
    {
      do
      {
        v44 = *v33;
        v34 = sub_CB72A0();
        v35 = (__int64)v34;
        if ( v44 )
          sub_2E39560(v44, (__int64)v34);
        else
          sub_904010((__int64)v34, "nullptr");
        ++v33;
        sub_904010(v35, ", ");
      }
      while ( (__int64 *)v42 != v33 );
    }
    v36 = sub_CB72A0();
    sub_904010((__int64)v36, "\n\tComputed roots: ");
    v37 = v46;
    v43 = &v46[v47];
    if ( v46 != v43 )
    {
      do
      {
        v45 = *v37;
        v38 = sub_CB72A0();
        v39 = (__int64)v38;
        if ( v45 )
          sub_2E39560(v45, (__int64)v38);
        else
          sub_904010((__int64)v38, "nullptr");
        ++v37;
        sub_904010(v39, ", ");
      }
      while ( v43 != v37 );
    }
    v40 = sub_CB72A0();
    sub_904010((__int64)v40, "\n");
    v41 = (__int64 *)sub_CB72A0();
    if ( v41[4] != v41[2] )
      sub_CB5AE0(v41);
    if ( v46 != (__int64 *)v48 )
      _libc_free((unsigned __int64)v46);
  }
LABEL_3:
  v6 = v51;
  v7 = (unsigned __int64)&v51[56 * (unsigned int)v52];
  if ( v51 != (_BYTE *)v7 )
  {
    do
    {
      v7 -= 56LL;
      v8 = *(_QWORD *)(v7 + 24);
      if ( v8 != v7 + 40 )
        _libc_free(v8);
    }
    while ( v6 != (_BYTE *)v7 );
    v7 = (unsigned __int64)v51;
  }
  if ( (_BYTE *)v7 != v53 )
    _libc_free(v7);
  if ( (_QWORD *)v49[0] != v50 )
    _libc_free(v49[0]);
  return v5;
}
