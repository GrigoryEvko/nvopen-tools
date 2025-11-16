// Function: sub_12A9750
// Address: 0x12a9750
//
__int64 __fastcall sub_12A9750(_QWORD *a1, int a2, int a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r15
  char *v13; // rax
  _QWORD *v14; // r14
  char *v15; // rax
  __int64 v16; // r10
  __int64 v17; // rax
  _QWORD *v18; // r12
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  _QWORD *v23; // rdx
  __int64 v24; // rsi
  char *v25; // rax
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 result; // rax
  unsigned int v29; // esi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // r8
  __int64 *v34; // r15
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // r15
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // r10
  __int64 *v43; // r12
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdx
  _QWORD *v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rsi
  _QWORD *v54; // rdx
  __int64 v55; // rsi
  char *v56; // rax
  __int64 v57; // rax
  __int64 v58; // [rsp+0h] [rbp-E0h]
  __int64 v59; // [rsp+8h] [rbp-D8h]
  __int64 v60; // [rsp+10h] [rbp-D0h]
  unsigned __int64 *v61; // [rsp+10h] [rbp-D0h]
  __int64 v62; // [rsp+10h] [rbp-D0h]
  __int64 v63; // [rsp+10h] [rbp-D0h]
  __int64 v64; // [rsp+10h] [rbp-D0h]
  __int64 v65; // [rsp+10h] [rbp-D0h]
  __int64 v67; // [rsp+20h] [rbp-C0h]
  __int64 v68; // [rsp+28h] [rbp-B8h]
  __int64 v69; // [rsp+28h] [rbp-B8h]
  __int64 v70; // [rsp+28h] [rbp-B8h]
  __int64 v71; // [rsp+28h] [rbp-B8h]
  unsigned __int64 *v72; // [rsp+30h] [rbp-B0h]
  _QWORD *v75; // [rsp+48h] [rbp-98h] BYREF
  _QWORD v76[2]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v77; // [rsp+60h] [rbp-80h]
  _BYTE v78[16]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v79; // [rsp+80h] [rbp-60h]
  _QWORD v80[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v81; // [rsp+A0h] [rbp-40h]

  v8 = *(_QWORD *)(a4 + 16);
  v9 = a1[5];
  v10 = v8;
  if ( a2 == 14 )
    v10 = *(_QWORD *)(v8 + 16);
  v11 = *(_QWORD *)(v10 + 16);
  v59 = v10;
  v67 = *(_QWORD *)(v11 + 16);
  v68 = sub_1643370(v9);
  v12 = sub_1646BA0(v68, 0);
  v79 = 257;
  v13 = sub_128F980((__int64)a1, v8);
  v14 = v13;
  if ( v12 != *(_QWORD *)v13 )
  {
    if ( (unsigned __int8)v13[16] > 0x10u )
    {
      v81 = 257;
      v49 = sub_15FDFF0(v13, v12, v80, 0);
      v50 = a1[7];
      v14 = (_QWORD *)v49;
      if ( v50 )
      {
        v72 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v50 + 40, v49);
        v51 = *v72;
        v52 = v14[3] & 7LL;
        v14[4] = v72;
        v51 &= 0xFFFFFFFFFFFFFFF8LL;
        v14[3] = v51 | v52;
        *(_QWORD *)(v51 + 8) = v14 + 3;
        *v72 = *v72 & 7 | (unsigned __int64)(v14 + 3);
      }
      sub_164B780(v14, v78);
      v53 = a1[6];
      if ( v53 )
      {
        v76[0] = a1[6];
        sub_1623A60(v76, v53, 2);
        v54 = v14 + 6;
        if ( v14[6] )
        {
          sub_161E7C0(v14 + 6);
          v54 = v14 + 6;
        }
        v55 = v76[0];
        v14[6] = v76[0];
        if ( v55 )
          sub_1623210(v76, v55, v54);
      }
    }
    else
    {
      v14 = (_QWORD *)sub_15A4A70(v13, v12);
    }
  }
  v79 = 257;
  v77 = 257;
  v15 = sub_128F980((__int64)a1, v11);
  v16 = (__int64)v15;
  if ( v12 != *(_QWORD *)v15 )
  {
    if ( (unsigned __int8)v15[16] > 0x10u )
    {
      v81 = 257;
      v40 = sub_15FDFF0(v15, v12, v80, 0);
      v41 = a1[7];
      v42 = v40;
      if ( v41 )
      {
        v43 = (__int64 *)a1[8];
        v62 = v40;
        sub_157E9D0(v41 + 40, v40);
        v42 = v62;
        v44 = *v43;
        v45 = *(_QWORD *)(v62 + 24);
        *(_QWORD *)(v62 + 32) = v43;
        v44 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v62 + 24) = v44 | v45 & 7;
        *(_QWORD *)(v44 + 8) = v62 + 24;
        *v43 = *v43 & 7 | (v62 + 24);
      }
      v63 = v42;
      sub_164B780(v42, v78);
      v46 = a1[6];
      v16 = v63;
      if ( v46 )
      {
        v75 = (_QWORD *)a1[6];
        sub_1623A60(&v75, v46, 2);
        v16 = v63;
        v47 = v63 + 48;
        if ( *(_QWORD *)(v63 + 48) )
        {
          v58 = v63;
          v64 = v63 + 48;
          sub_161E7C0(v64);
          v16 = v58;
          v47 = v64;
        }
        v48 = v75;
        *(_QWORD *)(v16 + 48) = v75;
        if ( v48 )
        {
          v65 = v16;
          sub_1623210(&v75, v48, v47);
          v16 = v65;
        }
      }
    }
    else
    {
      v16 = sub_15A4A70(v15, v12);
    }
  }
  v60 = v16;
  v17 = sub_1648A60(64, 1);
  v18 = (_QWORD *)v17;
  if ( v17 )
    sub_15F9210(v17, v68, v60, 0, 0, 0);
  v19 = a1[7];
  if ( v19 )
  {
    v61 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v19 + 40, v18);
    v20 = *v61;
    v21 = v18[3] & 7LL;
    v18[4] = v61;
    v20 &= 0xFFFFFFFFFFFFFFF8LL;
    v18[3] = v20 | v21;
    *(_QWORD *)(v20 + 8) = v18 + 3;
    *v61 = *v61 & 7 | (unsigned __int64)(v18 + 3);
  }
  sub_164B780(v18, v76);
  v22 = a1[6];
  if ( v22 )
  {
    v80[0] = a1[6];
    sub_1623A60(v80, v22, 2);
    v23 = v18 + 6;
    if ( v18[6] )
    {
      sub_161E7C0(v18 + 6);
      v23 = v18 + 6;
    }
    v24 = v80[0];
    v18[6] = v80[0];
    if ( v24 )
      sub_1623210(v80, v24, v23);
  }
  v75 = v18;
  if ( a2 == 14 )
  {
    v79 = 257;
    v81 = 257;
    v56 = sub_128F980((__int64)a1, v59);
    v57 = sub_12A95D0(a1 + 6, (__int64)v56, v12, (__int64)v80);
    v75 = sub_12A93B0(a1 + 6, v68, v57, (__int64)v78);
  }
  v79 = 257;
  v25 = sub_128F980((__int64)a1, v67);
  v26 = (__int64)v25;
  if ( v12 != *(_QWORD *)v25 )
  {
    if ( (unsigned __int8)v25[16] > 0x10u )
    {
      v81 = 257;
      v31 = sub_15FDFF0(v25, v12, v80, 0);
      v32 = a1[7];
      v33 = v31;
      if ( v32 )
      {
        v34 = (__int64 *)a1[8];
        v69 = v31;
        sub_157E9D0(v32 + 40, v31);
        v33 = v69;
        v35 = *v34;
        v36 = *(_QWORD *)(v69 + 24);
        *(_QWORD *)(v69 + 32) = v34;
        v35 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v69 + 24) = v35 | v36 & 7;
        *(_QWORD *)(v35 + 8) = v69 + 24;
        *v34 = *v34 & 7 | (v69 + 24);
      }
      v70 = v33;
      sub_164B780(v33, v78);
      v37 = a1[6];
      v26 = v70;
      if ( v37 )
      {
        v76[0] = a1[6];
        sub_1623A60(v76, v37, 2);
        v26 = v70;
        v38 = v70 + 48;
        if ( *(_QWORD *)(v70 + 48) )
        {
          sub_161E7C0(v70 + 48);
          v26 = v70;
        }
        v39 = v76[0];
        *(_QWORD *)(v26 + 48) = v76[0];
        if ( v39 )
        {
          v71 = v26;
          sub_1623210(v76, v39, v38);
          v26 = v71;
        }
      }
    }
    else
    {
      v26 = sub_15A4A70(v25, v12);
    }
  }
  *a5 = v26;
  if ( a3 )
  {
    v29 = (unsigned __int8)a2 << 16;
    LOBYTE(v29) = (16 * a3) | 5;
    v30 = sub_1643350(a1[5]);
    v80[0] = sub_159C470(v30, v29, 0);
    sub_12A9700(a6, v80);
    v27 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v27 < *(_DWORD *)(a6 + 12) )
      goto LABEL_25;
  }
  else
  {
    v27 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v27 < *(_DWORD *)(a6 + 12) )
      goto LABEL_25;
  }
  sub_16CD150(a6, a6 + 16, 0, 8);
  v27 = *(unsigned int *)(a6 + 8);
LABEL_25:
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v27) = v14;
  result = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
  *(_DWORD *)(a6 + 8) = result;
  if ( a2 == 14 )
  {
    sub_12A9700(a6, &v75);
    result = *(unsigned int *)(a6 + 8);
    if ( *(_DWORD *)(a6 + 12) > (unsigned int)result )
      goto LABEL_27;
LABEL_52:
    sub_16CD150(a6, a6 + 16, 0, 8);
    result = *(unsigned int *)(a6 + 8);
    goto LABEL_27;
  }
  if ( *(_DWORD *)(a6 + 12) <= (unsigned int)result )
    goto LABEL_52;
LABEL_27:
  *(_QWORD *)(*(_QWORD *)a6 + 8 * result) = v18;
  ++*(_DWORD *)(a6 + 8);
  return result;
}
