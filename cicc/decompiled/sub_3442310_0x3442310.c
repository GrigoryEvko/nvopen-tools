// Function: sub_3442310
// Address: 0x3442310
//
__int64 __fastcall sub_3442310(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v3; // rax
  unsigned int v4; // ebx
  __int64 v5; // r13
  int v6; // eax
  unsigned int v7; // r8d
  __int64 v8; // r15
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 *v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned __int8 *v15; // r13
  unsigned int v16; // edx
  unsigned int v17; // r12d
  __int64 *v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // r8
  __int64 v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // r8
  unsigned int v24; // edx
  unsigned int v25; // ebx
  __int64 v26; // r9
  unsigned __int8 *v27; // r14
  __int64 v28; // rdi
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  unsigned __int8 **v31; // rax
  unsigned __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rax
  unsigned __int8 **v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int8 **v42; // rax
  unsigned __int64 v44; // rax
  _DWORD *v45; // rax
  unsigned int v46; // edx
  unsigned int v47; // edx
  unsigned __int8 *v48; // rax
  __int64 v49; // r11
  __int64 v50; // r8
  unsigned int v51; // edx
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // r9
  unsigned __int8 *v55; // rax
  unsigned int v56; // edx
  __int64 v57; // [rsp+18h] [rbp-C8h]
  __int64 v58; // [rsp+20h] [rbp-C0h]
  __int64 v59; // [rsp+20h] [rbp-C0h]
  __int64 v60; // [rsp+28h] [rbp-B8h]
  __int64 v61; // [rsp+28h] [rbp-B8h]
  unsigned int v62; // [rsp+30h] [rbp-B0h]
  unsigned int v63; // [rsp+30h] [rbp-B0h]
  __int64 v64; // [rsp+30h] [rbp-B0h]
  __int64 v65; // [rsp+30h] [rbp-B0h]
  unsigned int v66; // [rsp+30h] [rbp-B0h]
  _QWORD *v67; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v68; // [rsp+38h] [rbp-A8h]
  __int64 v69; // [rsp+38h] [rbp-A8h]
  __int64 v70; // [rsp+38h] [rbp-A8h]
  __int64 v71; // [rsp+40h] [rbp-A0h]
  unsigned int v72; // [rsp+40h] [rbp-A0h]
  __int64 v73; // [rsp+48h] [rbp-98h]
  __int64 v74; // [rsp+48h] [rbp-98h]
  __int64 v75; // [rsp+48h] [rbp-98h]
  unsigned int v76; // [rsp+50h] [rbp-90h]
  unsigned int v77; // [rsp+54h] [rbp-8Ch]
  unsigned __int8 *v78; // [rsp+58h] [rbp-88h]
  unsigned __int64 v79; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v80; // [rsp+88h] [rbp-58h]
  unsigned __int64 v81; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v82; // [rsp+98h] [rbp-48h]
  char v83; // [rsp+A0h] [rbp-40h]
  unsigned int v84; // [rsp+A4h] [rbp-3Ch]
  unsigned int v85; // [rsp+A8h] [rbp-38h]

  v3 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v4 = *(_DWORD *)(v3 + 32);
  v5 = v3 + 24;
  if ( v4 > 0x40 )
  {
    v6 = sub_C444A0(v3 + 24);
    v7 = 0;
    if ( v4 == v6 )
      return v7;
    v8 = *a1;
    v9 = sub_C444A0(v5);
    v10 = v9;
    if ( v9 == v4 - 1 )
      goto LABEL_4;
LABEL_23:
    v45 = *(_DWORD **)(v8 + 24);
    if ( *v45 <= (unsigned int)v10 )
      v10 = (unsigned int)*v45;
    *(double *)a3.m128i_i64 = sub_3719C00(&v81, v5, v10, 1);
    v27 = sub_34007B0(
            *(_QWORD *)v8,
            (__int64)&v81,
            *(_QWORD *)(v8 + 32),
            **(_DWORD **)(v8 + 16),
            *(_QWORD *)(*(_QWORD *)(v8 + 16) + 8LL),
            0,
            a3,
            0);
    v25 = v46;
    v71 = v46;
    v15 = sub_3400BD0(
            *(_QWORD *)v8,
            v85,
            *(_QWORD *)(v8 + 32),
            **(unsigned int **)(v8 + 8),
            *(_QWORD *)(*(_QWORD *)(v8 + 8) + 8LL),
            0,
            a3,
            0);
    v17 = v47;
    v73 = v47;
    v48 = sub_3400BD0(
            *(_QWORD *)v8,
            v84,
            *(_QWORD *)(v8 + 32),
            **(unsigned int **)(v8 + 8),
            *(_QWORD *)(*(_QWORD *)(v8 + 8) + 8LL),
            0,
            a3,
            0);
    v49 = *(_QWORD *)v8;
    v50 = *(_QWORD *)(v8 + 16);
    v77 = v51;
    v78 = v48;
    v52 = *(_QWORD *)(v8 + 32);
    v53 = **(_DWORD **)(v8 + 40);
    if ( !v83 )
    {
      v80 = **(_DWORD **)(v8 + 40);
      if ( v53 > 0x40 )
      {
        v59 = v52;
        v61 = v50;
        v65 = v49;
        sub_C43690((__int64)&v79, 0, 0);
        v52 = v59;
        v50 = v61;
        v49 = v65;
      }
      else
      {
        v79 = 0;
      }
      goto LABEL_29;
    }
    v80 = **(_DWORD **)(v8 + 40);
    v54 = 1LL << ((unsigned __int8)v53 - 1);
    if ( v53 > 0x40 )
    {
      v57 = 1LL << ((unsigned __int8)v53 - 1);
      v76 = v53 - 1;
      v58 = v52;
      v60 = v50;
      v64 = v49;
      sub_C43690((__int64)&v79, 0, 0);
      v49 = v64;
      v50 = v60;
      v52 = v58;
      v54 = v57;
      if ( v80 > 0x40 )
      {
        *(_QWORD *)(v79 + 8LL * (v76 >> 6)) |= v57;
LABEL_29:
        v55 = sub_34007B0(v49, (__int64)&v79, v52, *(_DWORD *)v50, *(_QWORD *)(v50 + 8), 0, a3, 0);
        v23 = (__int64)v55;
        v26 = v56;
        if ( v80 > 0x40 && v79 )
        {
          v62 = v56;
          v68 = v55;
          j_j___libc_free_0_0(v79);
          v26 = v62;
          v23 = (__int64)v68;
        }
        **(_BYTE **)(v8 + 48) |= v83;
        **(_BYTE **)(v8 + 56) |= v85 != 0;
        **(_BYTE **)(v8 + 64) |= v84 != 0;
        if ( v82 > 0x40 && v81 )
        {
          v63 = v26;
          v69 = v23;
          j_j___libc_free_0_0(v81);
          v26 = v63;
          v23 = v69;
        }
        goto LABEL_9;
      }
    }
    else
    {
      v79 = 0;
    }
    v79 |= v54;
    goto LABEL_29;
  }
  v7 = 0;
  if ( !*(_QWORD *)(v3 + 24) )
    return v7;
  v44 = *(_QWORD *)(v3 + 24);
  v8 = *a1;
  if ( v44 != 1 )
  {
    v10 = v4;
    if ( v44 )
    {
      _BitScanReverse64(&v44, v44);
      v10 = v4 - 64 + ((unsigned int)v44 ^ 0x3F);
    }
    goto LABEL_23;
  }
LABEL_4:
  v11 = *(__int64 **)(v8 + 8);
  v12 = *(_QWORD **)v8;
  v13 = *v11;
  v14 = v11[1];
  v81 = 0;
  v82 = 0;
  v15 = (unsigned __int8 *)sub_33F17F0(v12, 51, (__int64)&v81, v13, v14);
  v17 = v16;
  if ( v81 )
    sub_B91220((__int64)&v81, v81);
  v18 = *(__int64 **)(v8 + 16);
  v19 = *(_QWORD **)v8;
  v78 = v15;
  v20 = v18[1];
  v21 = *v18;
  v77 = v17;
  v81 = 0;
  v82 = 0;
  v22 = sub_33F17F0(v19, 51, (__int64)&v81, v21, v20);
  v23 = (__int64)v22;
  v25 = v24;
  if ( v81 )
  {
    v67 = v22;
    sub_B91220((__int64)&v81, v81);
    v23 = (__int64)v67;
  }
  v26 = v25;
  v27 = (unsigned __int8 *)v23;
LABEL_9:
  v28 = *(_QWORD *)(v8 + 72);
  v29 = v17 | v73 & 0xFFFFFFFF00000000LL;
  v30 = *(unsigned int *)(v28 + 8);
  if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v28 + 12) )
  {
    v66 = v26;
    v70 = v23;
    sub_C8D5F0(v28, (const void *)(v28 + 16), v30 + 1, 0x10u, v23, v26);
    v26 = v66;
    v23 = v70;
    v30 = *(unsigned int *)(v28 + 8);
  }
  v31 = (unsigned __int8 **)(*(_QWORD *)v28 + 16 * v30);
  *v31 = v15;
  v31[1] = (unsigned __int8 *)v29;
  v32 = v25 | v71 & 0xFFFFFFFF00000000LL;
  ++*(_DWORD *)(v28 + 8);
  v33 = *(_QWORD *)(v8 + 80);
  v34 = *(unsigned int *)(v33 + 8);
  if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 12) )
  {
    v72 = v26;
    v75 = v23;
    sub_C8D5F0(*(_QWORD *)(v8 + 80), (const void *)(v33 + 16), v34 + 1, 0x10u, v23, v26);
    v34 = *(unsigned int *)(v33 + 8);
    v26 = v72;
    v23 = v75;
  }
  v35 = (unsigned __int8 **)(*(_QWORD *)v33 + 16 * v34);
  *v35 = v27;
  v35[1] = (unsigned __int8 *)v32;
  ++*(_DWORD *)(v33 + 8);
  v36 = *(_QWORD *)(v8 + 88);
  v37 = (unsigned int)v26;
  v38 = *(unsigned int *)(v36 + 8);
  if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 12) )
  {
    v74 = v23;
    sub_C8D5F0(*(_QWORD *)(v8 + 88), (const void *)(v36 + 16), v38 + 1, 0x10u, v23, v26);
    v38 = *(unsigned int *)(v36 + 8);
    v23 = v74;
  }
  v39 = (__int64 *)(*(_QWORD *)v36 + 16 * v38);
  v39[1] = v37;
  *v39 = v23;
  ++*(_DWORD *)(v36 + 8);
  v40 = *(_QWORD *)(v8 + 96);
  v41 = *(unsigned int *)(v40 + 8);
  if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 12) )
  {
    sub_C8D5F0(*(_QWORD *)(v8 + 96), (const void *)(v40 + 16), v41 + 1, 0x10u, v23, v26);
    v41 = *(unsigned int *)(v40 + 8);
  }
  v42 = (unsigned __int8 **)(*(_QWORD *)v40 + 16 * v41);
  v7 = 1;
  v42[1] = (unsigned __int8 *)v77;
  *v42 = v78;
  ++*(_DWORD *)(v40 + 8);
  return v7;
}
