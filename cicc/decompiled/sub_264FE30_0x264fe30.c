// Function: sub_264FE30
// Address: 0x264fe30
//
__int64 *__fastcall sub_264FE30(__int64 a1, __int64 *a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v5; // r15
  _QWORD *v7; // rcx
  __int64 v8; // rax
  __int64 *v9; // rsi
  __int64 *v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r13
  int v13; // eax
  int *v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // rsi
  __int64 v20; // rdx
  __int64 **v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rsi
  char v24; // al
  _QWORD *v25; // rax
  _QWORD *v26; // r14
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // r12
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  _QWORD *v33; // r14
  int *v34; // r14
  int *v35; // r13
  __int64 v36; // rbx
  int *v37; // r15
  __int64 *result; // rax
  __int64 **v39; // r12
  __int64 **i; // rbx
  __int64 v41; // rdi
  __int64 **v42; // r12
  __int64 **j; // rbx
  __int64 v44; // rdi
  char v45; // bl
  int *v46; // r14
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // r14
  _QWORD *v52; // rax
  _QWORD *v53; // r13
  __int64 v54; // rax
  __int64 v55; // rax
  _QWORD *v56; // [rsp+8h] [rbp-148h]
  _QWORD *v57; // [rsp+10h] [rbp-140h]
  __int64 v58; // [rsp+10h] [rbp-140h]
  __int64 *v59; // [rsp+18h] [rbp-138h]
  __int64 **v61; // [rsp+38h] [rbp-118h]
  __int64 v62; // [rsp+40h] [rbp-110h]
  char v63; // [rsp+48h] [rbp-108h]
  int *v64; // [rsp+48h] [rbp-108h]
  __int64 **v65; // [rsp+48h] [rbp-108h]
  __int64 v66; // [rsp+48h] [rbp-108h]
  _QWORD *v69; // [rsp+60h] [rbp-F0h] BYREF
  volatile signed __int32 *v70; // [rsp+68h] [rbp-E8h]
  __int64 v71; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v72; // [rsp+78h] [rbp-D8h]
  unsigned int v73; // [rsp+88h] [rbp-C8h]
  __int64 v74; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+98h] [rbp-B8h]
  int *v76; // [rsp+A0h] [rbp-B0h]
  int *v77; // [rsp+A8h] [rbp-A8h]
  __int64 v78; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v79; // [rsp+B8h] [rbp-98h]
  int *v80; // [rsp+C0h] [rbp-90h]
  int *v81; // [rsp+C8h] [rbp-88h]
  _QWORD *v82; // [rsp+D0h] [rbp-80h] BYREF
  volatile signed __int32 *v83; // [rsp+D8h] [rbp-78h]
  int *v84; // [rsp+E0h] [rbp-70h]
  int *v85; // [rsp+E8h] [rbp-68h]
  __int64 v86; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v87; // [rsp+F8h] [rbp-58h]
  __int64 v88; // [rsp+100h] [rbp-50h]
  __int64 v89; // [rsp+108h] [rbp-48h]

  v5 = a1;
  v7 = (_QWORD *)*a2;
  v8 = *(_QWORD *)*a2;
  v9 = *(__int64 **)(*a2 + 8);
  v62 = v8;
  v10 = *(__int64 **)(a3 + 72);
  v11 = *(__int64 **)(a3 + 80);
  v59 = v9;
  if ( v10 == v11 )
  {
LABEL_25:
    v12 = 0;
  }
  else
  {
    while ( 1 )
    {
      v12 = *v10;
      if ( v9 == *(__int64 **)(*v10 + 8) )
        break;
      v10 += 2;
      if ( v11 == v10 )
        goto LABEL_25;
    }
  }
  v13 = *(_DWORD *)(a5 + 16);
  if ( !v13 && (_QWORD *)a5 != v7 + 3 )
  {
    sub_264A680(a5, (__int64)(v7 + 3));
    v7 = (_QWORD *)*a2;
    v13 = *(_DWORD *)(a5 + 16);
  }
  if ( v13 == *((_DWORD *)v7 + 10) )
  {
    *(_BYTE *)(a3 + 2) |= *((_BYTE *)v7 + 16);
    if ( v12 )
    {
      v14 = (int *)(*(_QWORD *)(a5 + 8) + 4LL * *(unsigned int *)(a5 + 24));
      sub_22B0690(&v74, (__int64 *)a5);
      sub_264C8F0(v12 + 24, v14, v15, v16, v17, v18, v74, v75, v76, v77);
      v19 = 0;
      *(_BYTE *)(v12 + 16) |= *(_BYTE *)(*a2 + 16);
      sub_264E780(*a2, 0, 1);
    }
    else
    {
      *(_QWORD *)*a2 = a3;
      sub_2647660((unsigned __int64 *)(a3 + 72), a2);
      v19 = (__int64 *)*a2;
      sub_2647750(v62, *a2);
    }
  }
  else
  {
    v45 = sub_26484B0(a1, a5);
    if ( v12 )
    {
      v46 = (int *)(*(_QWORD *)(a5 + 8) + 4LL * *(unsigned int *)(a5 + 24));
      sub_22B0690(&v78, (__int64 *)a5);
      sub_264C8F0(v12 + 24, v46, v47, v48, v49, v50, v78, v79, v80, v81);
      *(_BYTE *)(v12 + 16) |= v45;
    }
    else
    {
      v82 = 0;
      v51 = *a2;
      v52 = (_QWORD *)sub_22077B0(0x48u);
      v53 = v52;
      if ( v52 )
      {
        v52[1] = 0x100000001LL;
        *v52 = off_49D3C50;
        v54 = *(_QWORD *)(v51 + 8);
        v86 = 0;
        v66 = v54;
        v87 = 0;
        v88 = 0;
        v89 = 0;
        sub_264A680((__int64)&v86, a5);
        *((_BYTE *)v53 + 32) = v45;
        *((_BYTE *)v53 + 33) = 0;
        v53[5] = 0;
        v53[2] = a3;
        v53[6] = 0;
        v53[3] = v66;
        v53[7] = 0;
        *((_DWORD *)v53 + 16) = 0;
        sub_2649AA0((__int64)(v53 + 5), (__int64)&v86);
        sub_2342640((__int64)&v86);
      }
      v55 = *a2;
      v83 = (volatile signed __int32 *)v53;
      v82 = v53 + 2;
      sub_2647660((unsigned __int64 *)(*(_QWORD *)(v55 + 8) + 48LL), &v82);
      sub_2647660((unsigned __int64 *)(a3 + 72), &v82);
      if ( v83 )
        sub_A191D0(v83);
    }
    *(_BYTE *)(a3 + 2) |= v45;
    sub_2649AE0(*a2 + 24, a5);
    v19 = (__int64 *)(*a2 + 24);
    *(_BYTE *)(*a2 + 16) = sub_26484B0(a1, (__int64)v19);
  }
  v21 = *(__int64 ***)(v62 + 48);
  v61 = *(__int64 ***)(v62 + 56);
  if ( v61 != v21 )
  {
    while ( 1 )
    {
      v29 = *v21;
      v30 = **v21;
      if ( v62 != v30 )
        goto LABEL_13;
      v19 = v59;
      if ( (__int64 *)v62 == v59 )
      {
LABEL_21:
        v21 += 2;
        if ( v61 == v21 )
          break;
      }
      else
      {
        v30 = a3;
LABEL_13:
        v22 = a5;
        v23 = (__int64)(v29 + 3);
        if ( *((_DWORD *)v29 + 10) >= *(_DWORD *)(a5 + 16) )
        {
          v22 = (__int64)(v29 + 3);
          v23 = a5;
        }
        sub_264FCD0((__int64)&v71, v23, v22);
        sub_2649AE0((__int64)(*v21 + 3), (__int64)&v71);
        *((_BYTE *)*v21 + 16) = sub_26484B0(v5, (__int64)(*v21 + 3));
        if ( a4 || (v31 = *(_QWORD **)(a3 + 48), v32 = *(_QWORD **)(a3 + 56), v31 == v32) )
        {
LABEL_16:
          v24 = sub_26484B0(v5, (__int64)&v71);
          v69 = 0;
          v63 = v24;
          v25 = (_QWORD *)sub_22077B0(0x48u);
          v26 = v25;
          if ( v25 )
          {
            v86 = 0;
            v25[1] = 0x100000001LL;
            *v25 = off_49D3C50;
            v87 = 0;
            v88 = 0;
            v89 = 0;
            sub_264A680((__int64)&v86, (__int64)&v71);
            v27 = v87;
            v26[2] = v30;
            *((_BYTE *)v26 + 33) = 0;
            v26[6] = v27;
            v28 = v88;
            v26[3] = a3;
            v26[7] = v28;
            LODWORD(v28) = v89;
            *((_BYTE *)v26 + 32) = v63;
            *((_DWORD *)v26 + 16) = v28;
            v26[5] = 1;
            ++v86;
            v87 = 0;
            v88 = 0;
            LODWORD(v89) = 0;
            sub_C7D6A0(0, 0, 4);
          }
          v70 = (volatile signed __int32 *)v26;
          v69 = v26 + 2;
          sub_2647660((unsigned __int64 *)(a3 + 48), &v69);
          sub_2647660((unsigned __int64 *)(*v69 + 72LL), &v69);
          if ( v70 )
            sub_A191D0(v70);
          v19 = (__int64 *)(4LL * v73);
          sub_C7D6A0(v72, (__int64)v19, 4);
          goto LABEL_21;
        }
        while ( 1 )
        {
          v33 = (_QWORD *)*v31;
          if ( *(_QWORD *)*v31 == v30 )
            break;
          v31 += 2;
          if ( v32 == v31 )
            goto LABEL_16;
        }
        v57 = v33 + 3;
        v64 = (int *)(v72 + 4LL * v73);
        sub_22B0690(&v82, &v71);
        if ( v84 != v64 )
        {
          v56 = v33;
          v34 = v64;
          v35 = v84;
          v65 = v21;
          v36 = (__int64)v57;
          v58 = v5;
          v37 = v85;
          do
          {
            sub_22B6470((__int64)&v86, v36, v35);
            do
              ++v35;
            while ( v35 != v37 && (unsigned int)*v35 > 0xFFFFFFFD );
          }
          while ( v34 != v35 );
          v33 = v56;
          v21 = v65;
          v5 = v58;
        }
        v19 = &v71;
        v21 += 2;
        *((_BYTE *)v33 + 16) |= sub_26484B0(v5, (__int64)&v71);
        sub_2342640((__int64)&v71);
        if ( v61 == v21 )
          break;
      }
    }
  }
  result = (__int64 *)sub_2647F70(v62, (__int64)v19, v20);
  *(_BYTE *)(v62 + 2) = (_BYTE)result;
  if ( (_BYTE)qword_4FF3AA8 )
  {
    if ( (_BYTE)result )
      sub_264C780((_QWORD *)v62);
    if ( *(_BYTE *)(a3 + 2) )
      sub_264C780((_QWORD *)a3);
    v39 = *(__int64 ***)(v62 + 56);
    for ( i = *(__int64 ***)(v62 + 48); v39 != i; i += 2 )
    {
      v41 = **i;
      if ( *(_BYTE *)(v41 + 2) )
        sub_264C780((_QWORD *)v41);
    }
    result = (__int64 *)a3;
    v42 = *(__int64 ***)(a3 + 56);
    for ( j = *(__int64 ***)(a3 + 48); v42 != j; j += 2 )
    {
      result = *j;
      v44 = **j;
      if ( *(_BYTE *)(v44 + 2) )
        result = sub_264C780((_QWORD *)v44);
    }
  }
  return result;
}
