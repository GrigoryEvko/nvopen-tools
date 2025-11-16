// Function: sub_3211090
// Address: 0x3211090
//
__int64 __fastcall sub_3211090(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rbx
  __int64 *v19; // r13
  __int64 v20; // rdx
  unsigned __int64 *v21; // r13
  unsigned __int64 *v22; // r14
  unsigned __int64 *v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r13
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 *v30; // rdi
  void (*v31)(); // rax
  __int64 v32; // rax
  void (*v33)(); // rdx
  void (*v34)(); // rax
  void (*v35)(); // rax
  _QWORD v36[4]; // [rsp-58h] [rbp-58h] BYREF
  char v37; // [rsp-38h] [rbp-38h]
  char v38; // [rsp-37h] [rbp-37h]

  result = *(_QWORD *)(a1 + 8);
  if ( result && *(_BYTE *)(result + 782) )
  {
    sub_3200CF0(a1, 0);
    v6 = sub_31F8650(a1, 241, v3, v4, v5);
    sub_31F8980(a1, 241, v7, v8, v9);
    sub_31F8B90(a1, 241, v10, v11, v12);
    v13 = v6;
    sub_31F8740(a1, v6);
    if ( *(_DWORD *)(a1 + 1176) )
      sub_31FFCF0(a1, v6, v14, v15, v16);
    v18 = *(__int64 **)(a1 + 1088);
    v19 = &v18[2 * *(unsigned int *)(a1 + 1096)];
    while ( v19 != v18 )
    {
      while ( (*(_BYTE *)(*v18 + 32) & 0xF) == 1 || sub_B2FC80(*v18) )
      {
        v18 += 2;
        if ( v19 == v18 )
          goto LABEL_12;
      }
      v20 = v18[1];
      v13 = *v18;
      v18 += 2;
      sub_320A500((__int64 *)a1, v13, v20);
    }
LABEL_12:
    sub_3205E80(a1, v13, v14, v15, v16, v17);
    sub_32085C0(a1);
    v21 = *(unsigned __int64 **)(a1 + 1344);
    v22 = *(unsigned __int64 **)(a1 + 1352);
    *(_QWORD *)(a1 + 1336) = 0;
    v23 = v21;
    if ( v21 != v22 )
    {
      do
      {
        if ( (unsigned __int64 *)*v23 != v23 + 2 )
          j_j___libc_free_0(*v23);
        v23 += 5;
      }
      while ( v22 != v23 );
      *(_QWORD *)(a1 + 1352) = v21;
    }
    sub_3209370(a1);
    sub_3200CF0(a1, 0);
    if ( *(_QWORD *)(a1 + 1376) != *(_QWORD *)(a1 + 1368) )
    {
      v27 = sub_31F8650(a1, 241, v24, v25, v26);
      sub_3205D70(a1, (__int64 *)(a1 + 1368), v28, v29);
      sub_31F8740(a1, v27);
    }
    v30 = *(__int64 **)(a1 + 528);
    v31 = *(void (**)())(*v30 + 120);
    v38 = 1;
    v36[0] = "File index to string table offset subsection";
    v37 = 3;
    if ( v31 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v31)(v30, v36, 1);
      v30 = *(__int64 **)(a1 + 528);
    }
    v32 = *v30;
    v33 = *(void (**)())(*v30 + 808);
    if ( v33 != nullsub_110 )
    {
      ((void (__fastcall *)(__int64 *))v33)(v30);
      v30 = *(__int64 **)(a1 + 528);
      v32 = *v30;
    }
    v34 = *(void (**)())(v32 + 120);
    v38 = 1;
    v36[0] = "String table";
    v37 = 3;
    if ( v34 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v34)(v30, v36, 1);
      v30 = *(__int64 **)(a1 + 528);
    }
    v35 = *(void (**)())(*v30 + 800);
    if ( v35 != nullsub_109 )
      ((void (__fastcall *)(__int64 *))v35)(v30);
    sub_31F8FA0((_QWORD *)a1);
    if ( (unsigned int)sub_3707AB0(a1 + 632) )
      sub_31F7B70(a1);
    if ( *(_BYTE *)(a1 + 784) )
    {
      if ( (unsigned int)sub_3707AB0(a1 + 632) )
        sub_31F4810(a1);
    }
    return sub_31FA4C0(a1);
  }
  return result;
}
