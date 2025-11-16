// Function: sub_121C160
// Address: 0x121c160
//
__int64 __fastcall sub_121C160(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  _QWORD *v4; // r13
  __int64 v5; // r14
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v10; // rax
  _QWORD **v11; // r12
  __int64 v12; // rax
  _QWORD *v13; // rbx
  __int64 v14; // r15
  const void *v15; // r13
  size_t v16; // r14
  _QWORD *v17; // r12
  size_t v18; // rbx
  size_t v19; // rdx
  int v20; // eax
  __int64 v21; // rbx
  const void *v22; // r9
  size_t v23; // rcx
  _QWORD *v24; // r8
  size_t v25; // rbx
  size_t v26; // rdx
  int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // rax
  _BYTE *v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // r8
  __int64 v37; // r14
  _QWORD *v38; // r12
  __int64 v39; // rdi
  _QWORD *v40; // rcx
  __int64 v41; // rdi
  size_t v42; // rbx
  size_t v43; // r14
  size_t v44; // rdx
  int v45; // eax
  unsigned int v46; // edi
  __int64 v47; // rbx
  size_t v48; // [rsp+10h] [rbp-A0h]
  _QWORD *v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  __int64 v51; // [rsp+18h] [rbp-98h]
  _QWORD *v52; // [rsp+20h] [rbp-90h]
  _QWORD *v53; // [rsp+20h] [rbp-90h]
  __int64 v55[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v56; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v57[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v58; // [rsp+70h] [rbp-40h]

  v4 = a1;
  if ( *(_BYTE *)(a3 + 8) != 14 )
  {
    v58 = 259;
    v8 = 0;
    v57[0] = "global variable reference must have pointer type";
    sub_11FD800((__int64)(a1 + 22), a4, (__int64)v57, 1);
    return v8;
  }
  v5 = a2;
  v7 = sub_1209B90(*(_QWORD *)(a1[43] + 120LL), *(const void **)a2, *(_QWORD *)(a2 + 8));
  if ( v7
    || (v10 = sub_1212F00((__int64)(a1 + 137), a2), v52 = a1 + 138, a1 + 138 != (_QWORD *)v10)
    && (v7 = *(_QWORD *)(v10 + 64)) != 0 )
  {
    sub_8FD6D0((__int64)v55, "@", (_QWORD *)a2);
    v58 = 260;
    v57[0] = v55;
    v8 = sub_120A960((__int64)a1, a4, (__int64)v57, a3, v7);
    if ( (__int64 *)v55[0] != &v56 )
      j_j___libc_free_0(v55[0], v56 + 1);
    return v8;
  }
  v11 = (_QWORD **)a1[43];
  v12 = sub_BCB2B0(*v11);
  BYTE4(v55[0]) = 1;
  v13 = (_QWORD *)v12;
  v58 = 257;
  LODWORD(v55[0]) = *(_DWORD *)(a3 + 8) >> 8;
  v8 = (__int64)sub_BD2C40(88, unk_3F0FAE8);
  if ( v8 )
    sub_B30000(v8, (__int64)v11, v13, 0, 9, 0, (__int64)v57, 0, 0, v55[0], 0);
  if ( !a1[139] )
  {
    v24 = a1 + 138;
    goto LABEL_32;
  }
  v50 = v8;
  v14 = a1[139];
  v15 = *(const void **)a2;
  v16 = *(_QWORD *)(a2 + 8);
  v17 = a1 + 138;
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v14 + 40);
      v19 = v16;
      if ( v18 <= v16 )
        v19 = *(_QWORD *)(v14 + 40);
      if ( v19 )
      {
        v20 = memcmp(*(const void **)(v14 + 32), v15, v19);
        if ( v20 )
          break;
      }
      v21 = v18 - v16;
      if ( v21 >= 0x80000000LL )
        goto LABEL_22;
      if ( v21 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v20 = v21;
        break;
      }
LABEL_13:
      v14 = *(_QWORD *)(v14 + 24);
      if ( !v14 )
        goto LABEL_23;
    }
    if ( v20 < 0 )
      goto LABEL_13;
LABEL_22:
    v17 = (_QWORD *)v14;
    v14 = *(_QWORD *)(v14 + 16);
  }
  while ( v14 );
LABEL_23:
  v22 = v15;
  v23 = v16;
  v8 = v50;
  v24 = v17;
  v4 = a1;
  v5 = a2;
  if ( v52 == v17 )
    goto LABEL_32;
  v25 = v17[5];
  v26 = v23;
  if ( v25 <= v23 )
    v26 = v17[5];
  if ( v26 && (v48 = v23, v27 = memcmp(v22, (const void *)v17[4], v26), v23 = v48, v24 = v17, v27) )
  {
LABEL_31:
    if ( v27 < 0 )
      goto LABEL_32;
  }
  else
  {
    v28 = v23 - v25;
    if ( v28 <= 0x7FFFFFFF )
    {
      if ( v28 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v27 = v28;
        goto LABEL_31;
      }
LABEL_32:
      v49 = v24;
      v29 = sub_22077B0(80);
      v30 = *(_BYTE **)v5;
      v31 = *(_QWORD *)(v5 + 8);
      v32 = v29 + 48;
      v33 = v29 + 32;
      v51 = v29;
      *(_QWORD *)(v29 + 32) = v29 + 48;
      sub_12060D0((__int64 *)(v29 + 32), v30, (__int64)&v30[v31]);
      *(_QWORD *)(v51 + 64) = 0;
      *(_QWORD *)(v51 + 72) = 0;
      v34 = sub_121BED0(a1 + 137, v49, v33);
      v36 = v51;
      v37 = v34;
      v38 = v35;
      if ( v35 )
      {
        if ( v34 || v52 == v35 )
        {
LABEL_35:
          v39 = 1;
          goto LABEL_36;
        }
        v42 = *(_QWORD *)(v51 + 40);
        v44 = v35[5];
        v43 = v44;
        if ( v42 <= v44 )
          v44 = *(_QWORD *)(v51 + 40);
        if ( v44 && (v45 = memcmp(*(const void **)(v51 + 32), (const void *)v38[4], v44), v36 = v51, (v46 = v45) != 0) )
        {
LABEL_49:
          v39 = v46 >> 31;
        }
        else
        {
          v47 = v42 - v43;
          v39 = 0;
          if ( v47 <= 0x7FFFFFFF )
          {
            if ( v47 < (__int64)0xFFFFFFFF80000000LL )
              goto LABEL_35;
            v46 = v47;
            goto LABEL_49;
          }
        }
LABEL_36:
        v40 = v52;
        v53 = (_QWORD *)v36;
        sub_220F040(v39, v36, v38, v40);
        ++v4[142];
        v24 = v53;
      }
      else
      {
        v41 = *(_QWORD *)(v51 + 32);
        if ( v32 != v41 )
        {
          j_j___libc_free_0(v41, *(_QWORD *)(v51 + 48) + 1LL);
          v36 = v51;
        }
        j_j___libc_free_0(v36, 80);
        v24 = (_QWORD *)v37;
      }
    }
  }
  v24[8] = v8;
  v24[9] = a4;
  return v8;
}
