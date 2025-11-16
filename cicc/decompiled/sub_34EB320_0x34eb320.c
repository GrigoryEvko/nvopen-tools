// Function: sub_34EB320
// Address: 0x34eb320
//
void __fastcall sub_34EB320(__int64 a1, char *a2, __int64 a3, char a4)
{
  _BYTE *v4; // r12
  char *v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 *v10; // r13
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // r10
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // rax
  unsigned __int64 *v19; // r10
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r13
  const void *v23; // r10
  __int64 v24; // r14
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 *v28; // r14
  __int64 v29; // rsi
  unsigned int v30; // r12d
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  _QWORD *v34; // rax
  int v35; // r8d
  unsigned __int64 v36; // rsi
  const void *v37; // r15
  size_t v38; // r14
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 *v41; // rdi
  int v42; // eax
  char v43; // dl
  char v44; // dl
  __int64 *v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // r14
  __int64 v48; // rbx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  char v52; // al
  _BYTE *v53; // [rsp+8h] [rbp-A8h]
  __int64 v54; // [rsp+10h] [rbp-A0h]
  _BYTE *dest; // [rsp+18h] [rbp-98h]
  __int64 *desta; // [rsp+18h] [rbp-98h]
  __int64 *v57; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v58; // [rsp+20h] [rbp-90h]
  unsigned int v59; // [rsp+20h] [rbp-90h]
  const void *v60; // [rsp+20h] [rbp-90h]
  unsigned __int64 v61; // [rsp+20h] [rbp-90h]
  __int64 *v65; // [rsp+38h] [rbp-78h]
  __int64 v66; // [rsp+48h] [rbp-68h] BYREF
  __int64 *v67; // [rsp+50h] [rbp-60h] BYREF
  __int64 v68; // [rsp+58h] [rbp-58h]
  _BYTE v69[80]; // [rsp+60h] [rbp-50h] BYREF

  v4 = (_BYTE *)a3;
  v5 = a2;
  v6 = *(_QWORD *)(a3 + 16);
  v7 = v6 + 48;
  if ( (unsigned __int8)sub_2E31AC0(v6) )
  {
    v8 = *(_QWORD *)(v6 + 56);
    if ( v8 != v7 )
    {
      dest = v4;
      v9 = v6 + 48;
      do
      {
        while ( 1 )
        {
          if ( *(_WORD *)(v8 + 68) == 2 )
          {
            v46 = *(_QWORD *)(v8 + 32);
            v47 = v46 + 40LL * (*(_DWORD *)(v8 + 40) & 0xFFFFFF);
            if ( v47 != v46 )
            {
              v48 = *(_QWORD *)(v8 + 32);
              do
              {
                if ( *(_BYTE *)v48 == 4 && !sub_2E322C0(*((_QWORD *)a2 + 2), *(_QWORD *)(v48 + 24)) )
                  sub_2E33F80(*((_QWORD *)a2 + 2), *(_QWORD *)(v48 + 24), 0, v49, v50, v51);
                v48 += 40;
              }
              while ( v47 != v48 );
            }
          }
          if ( (*(_BYTE *)v8 & 4) == 0 )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          if ( v8 == v9 )
            goto LABEL_8;
        }
        while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
        v8 = *(_QWORD *)(v8 + 8);
      }
      while ( v8 != v9 );
LABEL_8:
      v7 = v6 + 48;
      v5 = a2;
      v4 = dest;
    }
  }
  v10 = (__int64 *)sub_2E313E0(v6);
  v13 = sub_2E313E0(*((_QWORD *)v5 + 2));
  if ( v10 != *(__int64 **)(v6 + 56) && (__int64 *)v13 != v10 )
  {
    v57 = (__int64 *)v13;
    desta = *(__int64 **)(v6 + 56);
    sub_2E310C0((__int64 *)(*((_QWORD *)v5 + 2) + 40LL), (__int64 *)(v6 + 40), (__int64)desta, (__int64)v10);
    v13 = (unsigned __int64)v57;
    if ( v10 != v57 )
    {
      v14 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*desta & 0xFFFFFFFFFFFFFFF8LL) + 8) = v10;
      *v10 = *v10 & 7 | *desta & 0xFFFFFFFFFFFFFFF8LL;
      v15 = *v57;
      *(_QWORD *)(v14 + 8) = v57;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      *desta = v15 | *desta & 7;
      *(_QWORD *)(v15 + 8) = desta;
      *v57 = v14 | *v57 & 7;
    }
  }
  if ( v10 != (__int64 *)v7 )
  {
    v16 = *(_QWORD *)(a1 + 528);
    v17 = *(__int64 (**)())(*(_QWORD *)v16 + 920LL);
    if ( v17 == sub_2DB1B30
      || (v61 = v13,
          v52 = ((__int64 (__fastcall *)(__int64, __int64 *))v17)(v16, v10),
          v19 = (unsigned __int64 *)v61,
          !v52) )
    {
      v18 = *((_QWORD *)v5 + 2);
      v19 = (unsigned __int64 *)(v18 + 48);
    }
    else
    {
      v18 = *((_QWORD *)v5 + 2);
    }
    if ( v19 != (unsigned __int64 *)v7 )
    {
      v58 = v19;
      sub_2E310C0((__int64 *)(v18 + 40), (__int64 *)(v6 + 40), (__int64)v10, v7);
      if ( v10 != (__int64 *)v7 )
      {
        v20 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v7;
        *(_QWORD *)(v6 + 48) = *(_QWORD *)(v6 + 48) & 7LL | *v10 & 0xFFFFFFFFFFFFFFF8LL;
        v21 = *v58;
        *(_QWORD *)(v20 + 8) = v58;
        v21 &= 0xFFFFFFFFFFFFFFF8LL;
        *v10 = v21 | *v10 & 7;
        *(_QWORD *)(v21 + 8) = v10;
        *v58 = v20 | *v58 & 7;
      }
    }
  }
  if ( (*v5 & 0x10) != 0 )
    sub_2E33470(*(unsigned int **)(*((_QWORD *)v5 + 2) + 144LL), *(unsigned int **)(*((_QWORD *)v5 + 2) + 152LL));
  v22 = *(unsigned int *)(v6 + 120);
  v23 = *(const void **)(v6 + 112);
  v67 = (__int64 *)v69;
  v24 = 8 * v22;
  v68 = 0x400000000LL;
  if ( v22 > 4 )
  {
    v60 = v23;
    sub_C8D5F0((__int64)&v67, v69, v22, 8u, v11, v12);
    v23 = v60;
    v45 = &v67[(unsigned int)v68];
  }
  else
  {
    if ( !v24 )
      goto LABEL_23;
    v45 = (__int64 *)v69;
  }
  memcpy(v45, v23, 8 * v22);
  LODWORD(v24) = v68;
LABEL_23:
  v25 = *(_QWORD *)(v6 + 32);
  v59 = 0;
  LODWORD(v68) = v24 + v22;
  v26 = (unsigned int)(v24 + v22);
  v27 = *(_QWORD *)(v6 + 8);
  if ( v27 == v25 + 320 )
    v27 = 0;
  if ( (*v4 & 0x40) == 0 )
    v27 = 0;
  if ( a4 )
  {
    if ( sub_2E322C0(*((_QWORD *)v5 + 2), v6) )
    {
      v59 = sub_2E441D0(*(_QWORD *)(a1 + 544), *((_QWORD *)v5 + 2), v6);
      sub_2E33650(*((_QWORD *)v5 + 2), v6);
    }
    v26 = (unsigned int)v68;
  }
  v28 = v67;
  v65 = &v67[v26];
  if ( v65 != v67 )
  {
    v53 = v4;
    while ( 1 )
    {
      while ( 1 )
      {
        v29 = *v28;
        v66 = v29;
        if ( v29 == v27 )
        {
          sub_2E33650(v6, v27);
          goto LABEL_31;
        }
        if ( a4 )
          break;
        sub_2E33650(v6, v29);
LABEL_31:
        if ( v65 == ++v28 )
          goto LABEL_38;
      }
      v30 = sub_2E441D0(*(_QWORD *)(a1 + 544), v6, v29);
      if ( v59 )
        v30 = (v59 * (unsigned __int64)v30 + 0x40000000) >> 31;
      sub_2E33650(v6, v66);
      if ( !sub_2E322C0(*((_QWORD *)v5 + 2), v66) )
      {
        sub_2E33F80(*((_QWORD *)v5 + 2), v66, v30, v31, v32, v33);
        goto LABEL_31;
      }
      v54 = *((_QWORD *)v5 + 2);
      sub_2E441D0(*(_QWORD *)(a1 + 544), v54, v66);
      ++v28;
      v34 = sub_34E6760(
              *(_QWORD **)(*((_QWORD *)v5 + 2) + 112LL),
              *(_QWORD *)(*((_QWORD *)v5 + 2) + 112LL) + 8LL * *(unsigned int *)(*((_QWORD *)v5 + 2) + 120LL),
              &v66);
      sub_2E32F90(v54, (__int64)v34, v35);
      if ( v65 == v28 )
      {
LABEL_38:
        v4 = v53;
        break;
      }
    }
  }
  v36 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 320LL) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 != v36 )
    sub_2E320F0((__int64 *)v6, v36);
  if ( (*v5 & 0x10) != 0 && (*v4 & 0x10) != 0 )
    sub_2E33470(*(unsigned int **)(*((_QWORD *)v5 + 2) + 144LL), *(unsigned int **)(*((_QWORD *)v5 + 2) + 152LL));
  v37 = (const void *)*((_QWORD *)v4 + 27);
  v38 = 40LL * *((unsigned int *)v4 + 56);
  v39 = *((unsigned int *)v4 + 56);
  v40 = *((unsigned int *)v5 + 56);
  if ( v39 + v40 > (unsigned __int64)*((unsigned int *)v5 + 57) )
  {
    sub_C8D5F0((__int64)(v5 + 216), v5 + 232, v39 + v40, 0x28u, v11, v12);
    v40 = *((unsigned int *)v5 + 56);
  }
  if ( v38 )
  {
    memcpy((void *)(*((_QWORD *)v5 + 27) + 40 * v40), v37, v38);
    LODWORD(v40) = *((_DWORD *)v5 + 56);
  }
  v41 = v67;
  *((_DWORD *)v5 + 56) = v40 + v39;
  v42 = *((_DWORD *)v4 + 1);
  *((_DWORD *)v4 + 56) = 0;
  *((_DWORD *)v5 + 1) += v42;
  *((_DWORD *)v5 + 2) += *((_DWORD *)v4 + 2);
  *((_DWORD *)v5 + 3) += *((_DWORD *)v4 + 3);
  v43 = v4[1];
  *((_DWORD *)v4 + 1) = 0;
  *((_QWORD *)v4 + 1) = 0;
  LOBYTE(v42) = (v5[1] | v43) & 2 | v5[1] & 0xFD;
  v44 = *v5;
  v5[1] = v42;
  *v5 = v44 & 0xBB | *v4 & 0x40;
  *v4 &= ~4u;
  if ( v41 != (__int64 *)v69 )
    _libc_free((unsigned __int64)v41);
}
