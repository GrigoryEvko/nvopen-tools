// Function: sub_112A580
// Address: 0x112a580
//
unsigned __int8 *__fastcall sub_112A580(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v7; // r8
  unsigned __int8 *result; // rax
  _QWORD *v9; // r12
  char v10; // bl
  char v11; // r15
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r15
  unsigned int **v15; // r10
  __int64 v16; // rax
  unsigned int *v17; // rdi
  __int64 v18; // r9
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned int *v21; // r12
  __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned int v24; // esi
  unsigned int *v25; // rax
  bool v26; // zf
  unsigned int *v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // r12
  unsigned int *v31; // rbx
  __int64 v32; // rdx
  unsigned int v33; // esi
  unsigned int **v34; // [rsp+8h] [rbp-100h]
  __int64 v35; // [rsp+8h] [rbp-100h]
  char v36; // [rsp+10h] [rbp-F8h]
  __int64 v37; // [rsp+10h] [rbp-F8h]
  unsigned int **v38; // [rsp+10h] [rbp-F8h]
  __int64 v39; // [rsp+10h] [rbp-F8h]
  __int64 v40; // [rsp+10h] [rbp-F8h]
  __int64 v41; // [rsp+10h] [rbp-F8h]
  __int64 v42; // [rsp+10h] [rbp-F8h]
  __int64 v43; // [rsp+10h] [rbp-F8h]
  __int64 v44; // [rsp+10h] [rbp-F8h]
  unsigned __int8 *v45; // [rsp+20h] [rbp-E8h] BYREF
  unsigned __int8 *v46; // [rsp+28h] [rbp-E0h] BYREF
  __int64 v47; // [rsp+30h] [rbp-D8h] BYREF
  __int64 v48; // [rsp+38h] [rbp-D0h] BYREF
  __int64 v49; // [rsp+40h] [rbp-C8h] BYREF
  _BYTE v50[32]; // [rsp+48h] [rbp-C0h] BYREF
  __int16 v51; // [rsp+68h] [rbp-A0h]
  _BYTE v52[32]; // [rsp+78h] [rbp-90h] BYREF
  __int16 v53; // [rsp+98h] [rbp-70h]
  _BYTE v54[32]; // [rsp+A8h] [rbp-60h] BYREF
  __int16 v55; // [rsp+C8h] [rbp-40h]

  v7 = sub_1116810(a1, a3, &v45, &v46, &v47, &v48, &v49);
  result = 0;
  if ( v7 )
  {
    v9 = (_QWORD *)(a4 + 24);
    v36 = sub_B532C0(v47 + 24, v9, *(_WORD *)(a2 + 2) & 0x3F);
    v10 = sub_B532C0(v48 + 24, v9, *(_WORD *)(a2 + 2) & 0x3F);
    v11 = sub_B532C0(v49 + 24, v9, *(_WORD *)(a2 + 2) & 0x3F);
    v12 = sub_ACD720(*(__int64 **)(*(_QWORD *)(a1 + 32) + 72LL));
    if ( v36 )
    {
      v15 = *(unsigned int ***)(a1 + 32);
      v51 = 257;
      v53 = 257;
      v38 = v15;
      v16 = sub_92B530(v15, 0x28u, (__int64)v45, v46, (__int64)v50);
      v17 = v38[10];
      v34 = v38;
      v39 = v16;
      v18 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64))(*(_QWORD *)v17 + 16LL))(
              v17,
              29,
              v12,
              v16);
      if ( !v18 )
      {
        v55 = 257;
        v41 = sub_B504D0(29, v12, v39, (__int64)v54, 0, 0);
        (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)v34[11]
                                                                                                 + 16LL))(
          v34[11],
          v41,
          v52,
          v34[7],
          v34[8]);
        v18 = v41;
        v25 = *v34;
        v26 = *v34 == &(*v34)[4 * *((unsigned int *)v34 + 2)];
        v35 = (__int64)&(*v34)[4 * *((unsigned int *)v34 + 2)];
        if ( !v26 )
        {
          v27 = v25;
          do
          {
            v28 = *((_QWORD *)v27 + 1);
            v29 = *v27;
            v27 += 4;
            v42 = v18;
            sub_B99FD0(v18, v29, v28);
            v18 = v42;
          }
          while ( (unsigned int *)v35 != v27 );
        }
      }
      v12 = v18;
      if ( !v10 )
      {
LABEL_4:
        if ( !v11 )
          return sub_F162A0(a1, a2, v12);
LABEL_7:
        v13 = *(_QWORD *)(a1 + 32);
        v53 = 257;
        v51 = 257;
        v37 = sub_92B530((unsigned int **)v13, 0x26u, (__int64)v45, v46, (__int64)v50);
        v14 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v13 + 80) + 16LL))(
                *(_QWORD *)(v13 + 80),
                29,
                v12,
                v37);
        if ( !v14 )
        {
          v55 = 257;
          v14 = sub_B504D0(29, v12, v37, (__int64)v54, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v13 + 88) + 16LL))(
            *(_QWORD *)(v13 + 88),
            v14,
            v52,
            *(_QWORD *)(v13 + 56),
            *(_QWORD *)(v13 + 64));
          v21 = *(unsigned int **)v13;
          v22 = *(_QWORD *)v13 + 16LL * *(unsigned int *)(v13 + 8);
          while ( (unsigned int *)v22 != v21 )
          {
            v23 = *((_QWORD *)v21 + 1);
            v24 = *v21;
            v21 += 4;
            sub_B99FD0(v14, v24, v23);
          }
        }
        v12 = v14;
        return sub_F162A0(a1, a2, v12);
      }
    }
    else if ( !v10 )
    {
      goto LABEL_4;
    }
    v19 = *(_QWORD *)(a1 + 32);
    v53 = 257;
    v51 = 257;
    v40 = sub_92B530((unsigned int **)v19, 0x20u, (__int64)v45, v46, (__int64)v50);
    v20 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v19 + 80) + 16LL))(
            *(_QWORD *)(v19 + 80),
            29,
            v12,
            v40);
    if ( !v20 )
    {
      v55 = 257;
      v43 = sub_B504D0(29, v12, v40, (__int64)v54, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v19 + 88) + 16LL))(
        *(_QWORD *)(v19 + 88),
        v43,
        v52,
        *(_QWORD *)(v19 + 56),
        *(_QWORD *)(v19 + 64));
      v20 = v43;
      v30 = *(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8);
      if ( *(_QWORD *)v19 != v30 )
      {
        v31 = *(unsigned int **)v19;
        do
        {
          v32 = *((_QWORD *)v31 + 1);
          v33 = *v31;
          v31 += 4;
          v44 = v20;
          sub_B99FD0(v20, v33, v32);
          v20 = v44;
        }
        while ( (unsigned int *)v30 != v31 );
      }
    }
    v12 = v20;
    if ( !v11 )
      return sub_F162A0(a1, a2, v12);
    goto LABEL_7;
  }
  return result;
}
