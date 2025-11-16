// Function: sub_121D710
// Address: 0x121d710
//
__int64 __fastcall sub_121D710(__int64 *a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // r13
  __int64 *v5; // r12
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // edx
  size_t v13; // rbx
  const void *v14; // r14
  const char *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  const char *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r14
  __int64 *v21; // r15
  __int64 v22; // r13
  size_t v23; // r12
  const void *v24; // r14
  size_t v25; // rbx
  size_t v26; // rdx
  int v27; // eax
  __int64 v28; // rbx
  const void *v29; // r11
  size_t v30; // r14
  size_t v31; // rbx
  size_t v32; // rdx
  int v33; // eax
  __int64 v34; // rax
  _BYTE *v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // rdi
  __int64 v42; // rdi
  size_t v43; // rbx
  size_t v44; // rcx
  size_t v45; // rdx
  int v46; // eax
  unsigned int v47; // edi
  __int64 v48; // rbx
  __int64 *v49; // [rsp+0h] [rbp-B0h]
  __int64 v50; // [rsp+8h] [rbp-A8h]
  size_t v51; // [rsp+8h] [rbp-A8h]
  __int64 v52; // [rsp+10h] [rbp-A0h]
  __int64 v53; // [rsp+18h] [rbp-98h]
  __int64 *v54; // [rsp+20h] [rbp-90h]
  __int64 v55; // [rsp+20h] [rbp-90h]
  __int64 v56; // [rsp+20h] [rbp-90h]
  _QWORD v58[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v59; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v60[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v61; // [rsp+70h] [rbp-40h]

  v4 = a2;
  v5 = a1;
  v7 = sub_1209B90(*(_QWORD *)(a1[1] + 112), *(const void **)a2, *(_QWORD *)(a2 + 8));
  if ( v7 )
  {
LABEL_2:
    v8 = *a1;
    sub_8FD6D0((__int64)v58, "%", (_QWORD *)a2);
    v61 = 260;
    v60[0] = v58;
    v7 = sub_120A960(v8, a4, (__int64)v60, a3, v7);
    if ( (__int64 *)v58[0] != &v59 )
      j_j___libc_free_0(v58[0], v59 + 1);
    return v7;
  }
  v10 = sub_1213220((__int64)(a1 + 2), a2);
  v54 = a1 + 3;
  if ( a1 + 3 != (__int64 *)v10 )
  {
    v11 = *(_QWORD *)(v10 + 64);
    if ( v11 )
    {
      v7 = v11;
      goto LABEL_2;
    }
  }
  v12 = *(unsigned __int8 *)(a3 + 8);
  if ( (_BYTE)v12 == 13 || v12 == 7 )
  {
    HIBYTE(v61) = 1;
    v17 = *a1;
    v18 = "invalid use of a non-first-class type";
    goto LABEL_15;
  }
  if ( (_BYTE)v12 == 8 )
  {
    v19 = a1[1];
    v60[0] = a2;
    v61 = 260;
    v20 = sub_B2BE50(v19);
    v53 = sub_22077B0(80);
    if ( v53 )
      sub_AA4D50(v53, v20, (__int64)v60, v19, 0);
  }
  else
  {
    v60[0] = a2;
    v61 = 260;
    v53 = sub_22077B0(40);
    if ( v53 )
      sub_B2BA90(v53, a3, (__int64)v60, 0, 0);
  }
  v13 = *(_QWORD *)(a2 + 8);
  v14 = *(const void **)a2;
  v15 = sub_BD5D20(v53);
  if ( v13 != v16 || v13 && memcmp(v15, v14, v13) )
  {
    HIBYTE(v61) = 1;
    v17 = *a1;
    v18 = "name is too long which can result in name collisions, consider making the name shorter or increasing -non-glob"
          "al-value-max-name-size";
LABEL_15:
    v60[0] = v18;
    LOBYTE(v61) = 3;
    sub_11FD800(v17 + 176, a4, (__int64)v60, 1);
    return v7;
  }
  if ( !a1[4] )
  {
    v21 = a1 + 3;
    goto LABEL_41;
  }
  v21 = a1 + 3;
  v22 = a1[4];
  v23 = *(_QWORD *)(a2 + 8);
  v24 = *(const void **)a2;
  do
  {
    while ( 1 )
    {
      v25 = *(_QWORD *)(v22 + 40);
      v26 = v23;
      if ( v25 <= v23 )
        v26 = *(_QWORD *)(v22 + 40);
      if ( v26 )
      {
        v27 = memcmp(*(const void **)(v22 + 32), v24, v26);
        if ( v27 )
          break;
      }
      v28 = v25 - v23;
      if ( v28 >= 0x80000000LL )
        goto LABEL_31;
      if ( v28 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v27 = v28;
        break;
      }
LABEL_22:
      v22 = *(_QWORD *)(v22 + 24);
      if ( !v22 )
        goto LABEL_32;
    }
    if ( v27 < 0 )
      goto LABEL_22;
LABEL_31:
    v21 = (__int64 *)v22;
    v22 = *(_QWORD *)(v22 + 16);
  }
  while ( v22 );
LABEL_32:
  v29 = v24;
  v4 = a2;
  v30 = v23;
  v5 = a1;
  if ( v54 == v21 )
    goto LABEL_41;
  v31 = v21[5];
  v32 = v30;
  if ( v31 <= v30 )
    v32 = v21[5];
  if ( v32 && (v33 = memcmp(v29, (const void *)v21[4], v32)) != 0 )
  {
LABEL_40:
    if ( v33 < 0 )
      goto LABEL_41;
  }
  else if ( (__int64)(v30 - v31) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v30 - v31) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v33 = v30 - v31;
      goto LABEL_40;
    }
LABEL_41:
    v49 = v21;
    v34 = sub_22077B0(80);
    v35 = *(_BYTE **)v4;
    v36 = *(_QWORD *)(v4 + 8);
    v37 = v34 + 48;
    v21 = (__int64 *)v34;
    *(_QWORD *)(v34 + 32) = v34 + 48;
    v50 = v34 + 32;
    sub_12060D0((__int64 *)(v34 + 32), v35, (__int64)&v35[v36]);
    v21[8] = 0;
    v21[9] = 0;
    v38 = sub_121D480(a1 + 2, v49, v50);
    v40 = v39;
    if ( v39 )
    {
      if ( v54 == (__int64 *)v39 || v38 )
      {
LABEL_44:
        v41 = 1;
        goto LABEL_45;
      }
      v43 = v21[5];
      v45 = *(_QWORD *)(v39 + 40);
      v44 = v45;
      if ( v43 <= v45 )
        v45 = v21[5];
      if ( v45
        && (v51 = v44,
            v52 = v40,
            v46 = memcmp((const void *)v21[4], *(const void **)(v40 + 32), v45),
            v40 = v52,
            v44 = v51,
            (v47 = v46) != 0) )
      {
LABEL_58:
        v41 = v47 >> 31;
      }
      else
      {
        v48 = v43 - v44;
        v41 = 0;
        if ( v48 <= 0x7FFFFFFF )
        {
          if ( v48 < (__int64)0xFFFFFFFF80000000LL )
            goto LABEL_44;
          v47 = v48;
          goto LABEL_58;
        }
      }
LABEL_45:
      sub_220F040(v41, v21, v40, v54);
      ++v5[7];
    }
    else
    {
      v42 = v21[4];
      if ( v37 != v42 )
      {
        v55 = v38;
        j_j___libc_free_0(v42, v21[6] + 1);
        v38 = v55;
      }
      v56 = v38;
      j_j___libc_free_0(v21, 80);
      v21 = (__int64 *)v56;
    }
  }
  v21[8] = v53;
  v21[9] = a4;
  return v53;
}
