// Function: sub_39AD750
// Address: 0x39ad750
//
__int64 __fastcall sub_39AD750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned int v12; // esi
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  int v16; // r9d
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  unsigned int v22; // r9d
  unsigned __int64 *v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rbx
  unsigned __int64 v31; // r14
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  int v34; // r9d
  unsigned __int64 v35; // r10
  unsigned __int64 v36; // r8
  __int64 v37; // rax
  unsigned __int64 *v38; // rax
  int v39; // eax
  int v40; // r9d
  __int64 v41; // rax
  __int64 v44; // [rsp+10h] [rbp-110h]
  __int64 v45; // [rsp+18h] [rbp-108h]
  unsigned __int64 v46; // [rsp+20h] [rbp-100h]
  unsigned __int64 v47; // [rsp+28h] [rbp-F8h]
  __int64 v48; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v49; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v50; // [rsp+38h] [rbp-E8h]
  int v51; // [rsp+48h] [rbp-D8h]
  unsigned int v52; // [rsp+48h] [rbp-D8h]
  unsigned int v53; // [rsp+48h] [rbp-D8h]
  __int64 v54; // [rsp+48h] [rbp-D8h]
  __int64 v55; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v56; // [rsp+58h] [rbp-C8h]
  __int64 v57; // [rsp+60h] [rbp-C0h]
  __int64 v58; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v59; // [rsp+70h] [rbp-B0h]
  __int64 v60; // [rsp+78h] [rbp-A8h]
  __int64 v61; // [rsp+80h] [rbp-A0h]
  unsigned int v62; // [rsp+88h] [rbp-98h]
  char v63; // [rsp+90h] [rbp-90h]
  unsigned int v64; // [rsp+94h] [rbp-8Ch]
  __int64 v65; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v66; // [rsp+A8h] [rbp-78h]
  __int64 v67; // [rsp+B0h] [rbp-70h]
  __int64 v68; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v69; // [rsp+C0h] [rbp-60h]
  __int64 v70; // [rsp+C8h] [rbp-58h]
  __int64 v71; // [rsp+D0h] [rbp-50h]
  unsigned int v72; // [rsp+D8h] [rbp-48h]
  char v73; // [rsp+E0h] [rbp-40h]
  unsigned int v74; // [rsp+E4h] [rbp-3Ch]

  result = a2 + 320;
  v7 = *(_QWORD *)(a2 + 328);
  v45 = a2 + 320;
  if ( v7 == a2 + 320 )
    return result;
  while ( 1 )
  {
    result = v45;
    v8 = v7;
    do
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( v8 == v45 )
      {
        if ( *(_BYTE *)(v7 + 184) )
          return result;
        if ( *(_QWORD *)(a2 + 328) == v7 )
          goto LABEL_25;
LABEL_7:
        v9 = sub_157ED20(*(_QWORD *)(v7 + 40));
        v10 = *(unsigned int *)(a3 + 56);
        v11 = *(_QWORD *)(a3 + 40);
        if ( (_DWORD)v10 )
        {
          v12 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v13 = (__int64 *)(v11 + 16LL * v12);
          v14 = *v13;
          if ( v9 == *v13 )
          {
LABEL_9:
            v51 = *((_DWORD *)v13 + 2);
            v15 = sub_39AC850(v7);
            v16 = v51;
            v17 = v15;
            goto LABEL_10;
          }
          v39 = 1;
          while ( v14 != -8 )
          {
            v40 = v39 + 1;
            v41 = ((_DWORD)v10 - 1) & (v12 + v39);
            v12 = v41;
            v13 = (__int64 *)(v11 + 16 * v41);
            v14 = *v13;
            if ( v9 == *v13 )
              goto LABEL_9;
            v39 = v40;
          }
        }
        v13 = (__int64 *)(v11 + 16 * v10);
        goto LABEL_9;
      }
    }
    while ( !*(_BYTE *)(v8 + 183) );
    if ( !*(_BYTE *)(v7 + 184) )
      break;
LABEL_22:
    v7 = v8;
  }
  if ( *(_QWORD *)(a2 + 328) != v7 )
    goto LABEL_7;
LABEL_25:
  v16 = -1;
  v17 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 384LL);
LABEL_10:
  v52 = v16;
  v18 = sub_39ACBF0(a1, v17);
  v19 = v52;
  v20 = v18;
  v21 = *(unsigned int *)(a4 + 8);
  v22 = v52;
  if ( (unsigned int)v21 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v52, v52);
    v21 = *(unsigned int *)(a4 + 8);
    v19 = v52;
    v22 = v52;
  }
  v23 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v21);
  v53 = v22;
  *v23 = v20;
  v23[1] = v19;
  ++*(_DWORD *)(a4 + 8);
  v24 = *(_QWORD *)v8;
  v25 = *(_QWORD *)(v7 + 32);
  v74 = v22;
  v72 = v22;
  v49 = v25;
  v69 = (v24 & 0xFFFFFFFFFFFFFFF8LL) + 24;
  v65 = a3;
  v66 = 0;
  v67 = v8;
  v68 = v8;
  v73 = 0;
  v70 = 0;
  v71 = 0;
  sub_39AC5C0((__int64)&v65);
  v55 = a3;
  v59 = v49;
  v56 = 0;
  v57 = v7;
  v58 = v8;
  v63 = 0;
  v64 = v53;
  v60 = 0;
  v61 = 0;
  v62 = v53;
  sub_39AC5C0((__int64)&v55);
  v26 = v57;
  v44 = v8;
  v48 = v66;
  v27 = v67;
  v67 = v57;
  v54 = v27;
  v50 = v69;
  v65 = v55;
  v66 = v56;
  v68 = v58;
  v69 = v59;
  v70 = v60;
  v71 = v61;
  v72 = v62;
  v73 = v63;
  v74 = v64;
  while ( 1 )
  {
    if ( v54 == v26 && v69 == v50 )
    {
      result = v48;
      if ( v66 == v48 )
        break;
    }
    v28 = *(_QWORD *)(a1 + 8);
    v29 = v71;
    v30 = *(_QWORD *)(v28 + 248);
    if ( !v71 )
      v29 = v70;
    v31 = sub_38CB470(1, *(_QWORD *)(v28 + 248));
    v32 = sub_39ACBF0(a1, v29);
    v33 = sub_38CB1F0(0, v32, v31, v30, 0);
    v35 = v72;
    v36 = v33;
    v37 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v37 >= *(_DWORD *)(a4 + 12) )
    {
      v46 = v72;
      v47 = v36;
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v36, v34);
      v37 = *(unsigned int *)(a4 + 8);
      v35 = v46;
      v36 = v47;
    }
    v38 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v37);
    *v38 = v36;
    v38[1] = v35;
    ++*(_DWORD *)(a4 + 8);
    sub_39AC5C0((__int64)&v65);
    v26 = v67;
  }
  v8 = v44;
  if ( v44 != v45 )
    goto LABEL_22;
  return result;
}
