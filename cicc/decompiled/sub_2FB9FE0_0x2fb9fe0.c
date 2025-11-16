// Function: sub_2FB9FE0
// Address: 0x2fb9fe0
//
unsigned __int64 __fastcall sub_2FB9FE0(__int64 *a1, int a2, int *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // r9
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rbx
  unsigned int v18; // esi
  __int64 v19; // r13
  __int64 v20; // r15
  __int64 *v21; // rdx
  __int64 v22; // r15
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r14
  unsigned __int64 v31; // rcx
  __int64 v32; // r9
  __int64 v34; // rdx
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 *v41; // r13
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // r15
  unsigned __int64 v45; // rax
  _QWORD *v46; // rdx
  _QWORD *v47; // rdi
  __int64 v48; // rbx
  unsigned __int64 v49; // r15
  _QWORD *v50; // rax
  _QWORD *v51; // rsi
  __int64 *v52; // [rsp+0h] [rbp-A0h]
  int v53; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v54; // [rsp+8h] [rbp-98h]
  __int32 v55; // [rsp+10h] [rbp-90h]
  _QWORD *v56; // [rsp+10h] [rbp-90h]
  int v57; // [rsp+10h] [rbp-90h]
  __int64 v58; // [rsp+10h] [rbp-90h]
  unsigned __int8 v63; // [rsp+38h] [rbp-68h]
  _QWORD *v64; // [rsp+38h] [rbp-68h]
  __int64 v65; // [rsp+38h] [rbp-68h]
  __int64 v66; // [rsp+48h] [rbp-58h]
  int *v67; // [rsp+50h] [rbp-50h] BYREF
  __int64 v68; // [rsp+58h] [rbp-48h]
  __int64 v69; // [rsp+60h] [rbp-40h]

  v7 = a1;
  v8 = a1[9];
  v9 = (unsigned int)(*(_DWORD *)(v8 + 64) + a2);
  v10 = a1[1];
  v11 = **(_QWORD **)(v8 + 16);
  v12 = *(unsigned int *)(v10 + 160);
  v13 = *(unsigned int *)(v11 + 4 * v9);
  v14 = *(_DWORD *)(v11 + 4 * v9) & 0x7FFFFFFF;
  v15 = v14;
  v16 = 8LL * v14;
  if ( v14 >= (unsigned int)v12 || (v17 = *(_QWORD *)(*(_QWORD *)(v10 + 152) + 8LL * v14)) == 0 )
  {
    v39 = v14 + 1;
    if ( (unsigned int)v12 < v39 && v39 != v12 )
    {
      if ( v39 >= v12 )
      {
        v48 = *(_QWORD *)(v10 + 168);
        v49 = v39 - v12;
        if ( v39 > (unsigned __int64)*(unsigned int *)(v10 + 164) )
        {
          v65 = a1[1];
          v57 = v13;
          sub_C8D5F0(v10 + 152, (const void *)(v10 + 168), v39, 8u, v10, v13);
          v10 = v65;
          LODWORD(v13) = v57;
          v12 = *(unsigned int *)(v65 + 160);
        }
        v40 = *(_QWORD *)(v10 + 152);
        v50 = (_QWORD *)(v40 + 8 * v12);
        v51 = &v50[v49];
        if ( v50 != v51 )
        {
          do
            *v50++ = v48;
          while ( v51 != v50 );
          LODWORD(v12) = *(_DWORD *)(v10 + 160);
          v40 = *(_QWORD *)(v10 + 152);
        }
        *(_DWORD *)(v10 + 160) = v49 + v12;
        goto LABEL_29;
      }
      *(_DWORD *)(v10 + 160) = v39;
    }
    v40 = *(_QWORD *)(v10 + 152);
LABEL_29:
    v41 = (__int64 *)(v40 + v16);
    v64 = (_QWORD *)v10;
    v42 = sub_2E10F30(v13);
    *v41 = v42;
    v17 = v42;
    sub_2E11E80(v64, v42);
    v10 = a1[1];
    v12 = *(unsigned int *)(v10 + 160);
    v13 = *(unsigned int *)(**(_QWORD **)(a1[9] + 16) + 4LL * (unsigned int)(*(_DWORD *)(a1[9] + 64) + a2));
    v14 = *(_DWORD *)(**(_QWORD **)(a1[9] + 16) + 4LL * (unsigned int)(*(_DWORD *)(a1[9] + 64) + a2)) & 0x7FFFFFFF;
    v15 = v14;
  }
  v63 = a2 != 0;
  v18 = *(_DWORD *)(*(_QWORD *)(a1[2] + 80) + 4 * v15);
  if ( v18 )
  {
    v13 = v18;
    v14 = v18 & 0x7FFFFFFF;
    v15 = v18 & 0x7FFFFFFF;
  }
  v19 = 8 * v15;
  if ( (unsigned int)v12 <= v14 || (v20 = *(_QWORD *)(*(_QWORD *)(v10 + 152) + 8 * v15)) == 0 )
  {
    v36 = v14 + 1;
    if ( v36 > (unsigned int)v12 )
    {
      v43 = v36;
      if ( v36 != v12 )
      {
        if ( v36 >= v12 )
        {
          v44 = *(_QWORD *)(v10 + 168);
          v45 = v36 - v12;
          if ( v43 > *(unsigned int *)(v10 + 164) )
          {
            v53 = v13;
            v54 = v45;
            v58 = v10;
            sub_C8D5F0(v10 + 152, (const void *)(v10 + 168), v43, 8u, v10, v13);
            v10 = v58;
            LODWORD(v13) = v53;
            v45 = v54;
            v12 = *(unsigned int *)(v58 + 160);
          }
          v37 = *(_QWORD *)(v10 + 152);
          v46 = (_QWORD *)(v37 + 8 * v12);
          v47 = &v46[v45];
          if ( v46 != v47 )
          {
            do
              *v46++ = v44;
            while ( v47 != v46 );
            LODWORD(v12) = *(_DWORD *)(v10 + 160);
            v37 = *(_QWORD *)(v10 + 152);
          }
          *(_DWORD *)(v10 + 160) = v45 + v12;
          goto LABEL_26;
        }
        *(_DWORD *)(v10 + 160) = v36;
      }
    }
    v37 = *(_QWORD *)(v10 + 152);
LABEL_26:
    v56 = (_QWORD *)v10;
    v38 = sub_2E10F30(v13);
    *(_QWORD *)(v37 + v19) = v38;
    v20 = v38;
    sub_2E11E80(v56, v38);
  }
  v21 = (__int64 *)sub_2E09D00((__int64 *)v20, a4);
  if ( v21 == (__int64 *)(*(_QWORD *)v20 + 24LL * *(unsigned int *)(v20 + 8)) )
  {
    v55 = *(_DWORD *)(v17 + 112);
  }
  else
  {
    v55 = *(_DWORD *)(v17 + 112);
    if ( (*(_DWORD *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v21 >> 1) & 3) <= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(a4 >> 1)
                                                                                             & 3) )
    {
      v34 = v21[2];
      if ( v34 )
      {
        v35 = v7[9];
        v68 = 0;
        v67 = a3;
        v68 = *(_QWORD *)((*(_QWORD *)(v34 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
        if ( (unsigned __int8)sub_350A430(v35, &v67, v34, a4, 1) )
        {
          if ( !(unsigned __int8)sub_2FB23E0((__int64)v7, v68, a5, a4) )
          {
            v31 = sub_350B670(v7[9], a5, (_DWORD)a6, v55, (unsigned int)&v67, v7[6], v63, 0, 0);
            return sub_2FB7970((__int64)v7, a2, a3, v31, 0, v32);
          }
        }
      }
    }
  }
  v22 = *(_QWORD *)(v20 + 104);
  if ( !v22 )
  {
    v25 = -1;
    v23 = -1;
LABEL_37:
    v31 = sub_2FB9C60(v7, *(unsigned int *)(*(_QWORD *)(v7[9] + 8) + 112LL), v55, v25, v23, a5, a6, v63, a2);
    return sub_2FB7970((__int64)v7, a2, a3, v31, 0, v32);
  }
  v52 = v7;
  v23 = 0;
  v24 = v22;
  v25 = 0;
  do
  {
    v26 = (__int64 *)sub_2E09D00((__int64 *)v24, a4);
    if ( v26 != (__int64 *)(*(_QWORD *)v24 + 24LL * *(unsigned int *)(v24 + 8))
      && (*(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v26 >> 1) & 3) <= ((unsigned int)(a4 >> 1)
                                                                                             & 3
                                                                                             | *(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
    {
      v25 |= *(_QWORD *)(v24 + 112);
      v23 |= *(_QWORD *)(v24 + 120);
    }
    v24 = *(_QWORD *)(v24 + 104);
  }
  while ( v24 );
  v7 = v52;
  if ( v23 | v25 )
    goto LABEL_37;
  v27 = v52[5];
  v66 = 0;
  v28 = *(_QWORD *)(v27 + 8);
  v67 = 0;
  v68 = 0;
  v69 = 0;
  sub_2F26260(a5, a6, (__int64 *)&v67, v28 - 400, v55);
  v30 = v29;
  if ( v67 )
    sub_B91220((__int64)&v67, (__int64)v67);
  v31 = sub_2E192D0(*(_QWORD *)(v52[1] + 32), v30, v63) & 0xFFFFFFFFFFFFFFF8LL | 4;
  return sub_2FB7970((__int64)v7, a2, a3, v31, 0, v32);
}
