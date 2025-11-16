// Function: sub_28C91C0
// Address: 0x28c91c0
//
__int64 __fastcall sub_28C91C0(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // rbx
  __int64 v7; // r13
  bool v8; // al
  __int64 v9; // r11
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r11
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r11
  __int64 v18; // rsi
  int v19; // r10d
  __int64 v20; // r14
  char *v21; // rbx
  char *v22; // rcx
  int v23; // r13d
  __int64 v24; // r12
  __int64 *v25; // rdi
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r9
  unsigned int v29; // r8d
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // rdx
  unsigned int v34; // r9d
  unsigned int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // eax
  unsigned int v42; // r8d
  int v43; // edx
  unsigned int v44; // r9d
  int v45; // edx
  int v46; // eax
  int v47; // r9d
  __int64 v48; // rdx
  __int64 *v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // r12
  __int64 v53; // r9
  __int64 *v54; // rbx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // rax
  int v58; // [rsp+4h] [rbp-5Ch]
  int v59; // [rsp+4h] [rbp-5Ch]
  int v60; // [rsp+4h] [rbp-5Ch]
  char *v61; // [rsp+8h] [rbp-58h]
  __int64 v62; // [rsp+10h] [rbp-50h]
  char *v63; // [rsp+18h] [rbp-48h]
  __int64 *v65; // [rsp+28h] [rbp-38h]

  result = a2 - (char *)a1;
  v63 = a2;
  v62 = a3;
  if ( a2 - (char *)a1 <= 256 )
    return result;
  if ( !a3 )
  {
    v65 = (__int64 *)a2;
LABEL_46:
    v51 = result >> 4;
    v52 = ((result >> 4) - 2) >> 1;
    sub_28C8760((__int64)a1, v52, result >> 4, a1[2 * v52], a1[2 * v52 + 1], a4);
    do
    {
      --v52;
      sub_28C8760((__int64)a1, v52, v51, a1[2 * v52], a1[2 * v52 + 1], v53);
    }
    while ( v52 );
    v54 = v65;
    do
    {
      v54 -= 2;
      v55 = *v54;
      v56 = v54[1];
      *v54 = *a1;
      v54[1] = a1[1];
      result = sub_28C8760((__int64)a1, 0, ((char *)v54 - (char *)a1) >> 4, v55, v56, v53);
    }
    while ( (char *)v54 - (char *)a1 > 16 );
    return result;
  }
  v61 = (char *)(a1 + 2);
  while ( 2 )
  {
    --v62;
    v6 = &a1[2 * (result >> 5)];
    v7 = v6[1];
    v8 = sub_28C85D0(a4, a1[3], v7);
    v10 = *a1;
    v11 = *((_QWORD *)v63 - 1);
    if ( !v8 )
    {
      if ( sub_28C85D0(a4, v9, v11) )
      {
        v49 = a1;
        v50 = a1[2];
LABEL_44:
        *v49 = v50;
        v16 = v49[1];
        v49[2] = v10;
        v49[1] = v17;
        v49[3] = v16;
        v18 = *((_QWORD *)v63 - 1);
        goto LABEL_7;
      }
      if ( !sub_28C85D0(a4, v7, v48) )
        goto LABEL_6;
LABEL_52:
      *a1 = *((_QWORD *)v63 - 2);
      v57 = *((_QWORD *)v63 - 1);
      *((_QWORD *)v63 - 2) = v10;
      v18 = a1[1];
      a1[1] = v57;
      *((_QWORD *)v63 - 1) = v18;
      v16 = a1[3];
      v17 = a1[1];
      goto LABEL_7;
    }
    if ( !sub_28C85D0(a4, v7, v11) )
    {
      if ( !sub_28C85D0(a4, v13, v12) )
      {
        v49 = a1;
        v50 = a1[2];
        goto LABEL_44;
      }
      goto LABEL_52;
    }
LABEL_6:
    *a1 = *v6;
    v14 = v6[1];
    *v6 = v10;
    v15 = a1[1];
    a1[1] = v14;
    v6[1] = v15;
    v16 = a1[3];
    v17 = a1[1];
    v18 = *((_QWORD *)v63 - 1);
LABEL_7:
    v19 = *(_DWORD *)(a4 + 2376);
    v20 = *(_QWORD *)(a4 + 2360);
    v21 = v61;
    v22 = v63;
    v23 = v19 - 1;
    while ( 1 )
    {
      v65 = (__int64 *)v21;
      v24 = v23 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v25 = (__int64 *)(v20 + 16 * v24);
      if ( !v19 )
        break;
      v26 = v23 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v27 = (__int64 *)(v20 + 16LL * v26);
      v28 = *v27;
      if ( v16 == *v27 )
      {
LABEL_11:
        v29 = *((_DWORD *)v27 + 2);
      }
      else
      {
        v46 = 1;
        while ( v28 != -4096 )
        {
          v26 = v23 & (v46 + v26);
          v60 = v46 + 1;
          v27 = (__int64 *)(v20 + 16LL * v26);
          v28 = *v27;
          if ( *v27 == v16 )
            goto LABEL_11;
          v46 = v60;
        }
        v29 = 0;
      }
      v30 = *v25;
      v31 = (__int64 *)(v20 + 16 * v24);
      if ( v17 != *v25 )
      {
        v44 = v23 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v45 = 1;
        while ( v30 != -4096 )
        {
          v44 = v23 & (v45 + v44);
          v59 = v45 + 1;
          v31 = (__int64 *)(v20 + 16LL * v44);
          v30 = *v31;
          if ( v17 == *v31 )
            goto LABEL_13;
          v45 = v59;
        }
        break;
      }
LABEL_13:
      if ( *((_DWORD *)v31 + 2) <= v29 )
        break;
LABEL_8:
      v16 = *((_QWORD *)v21 + 3);
      v21 += 16;
    }
    v22 -= 16;
    while ( v19 )
    {
      v32 = *v25;
      v33 = (__int64 *)(v20 + 16 * v24);
      if ( *v25 == v17 )
      {
LABEL_16:
        v34 = *((_DWORD *)v33 + 2);
      }
      else
      {
        v42 = v23 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v43 = 1;
        while ( v32 != -4096 )
        {
          v47 = v43 + 1;
          v42 = v23 & (v43 + v42);
          v33 = (__int64 *)(v20 + 16LL * v42);
          v32 = *v33;
          if ( v17 == *v33 )
            goto LABEL_16;
          v43 = v47;
        }
        v34 = 0;
      }
      v35 = v23 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v36 = (__int64 *)(v20 + 16LL * v35);
      v37 = *v36;
      if ( v18 != *v36 )
      {
        v41 = 1;
        while ( v37 != -4096 )
        {
          v35 = v23 & (v41 + v35);
          v58 = v41 + 1;
          v36 = (__int64 *)(v20 + 16LL * v35);
          v37 = *v36;
          if ( v18 == *v36 )
            goto LABEL_18;
          v41 = v58;
        }
        break;
      }
LABEL_18:
      if ( v34 >= *((_DWORD *)v36 + 2) )
        break;
      v18 = *((_QWORD *)v22 - 1);
      v22 -= 16;
    }
    if ( v21 < v22 )
    {
      v38 = *(_QWORD *)v21;
      *(_QWORD *)v21 = *(_QWORD *)v22;
      v39 = *((_QWORD *)v22 + 1);
      *(_QWORD *)v22 = v38;
      v40 = *((_QWORD *)v21 + 1);
      *((_QWORD *)v21 + 1) = v39;
      v18 = *((_QWORD *)v22 - 1);
      *((_QWORD *)v22 + 1) = v40;
      v19 = *(_DWORD *)(a4 + 2376);
      v20 = *(_QWORD *)(a4 + 2360);
      v17 = a1[1];
      v23 = v19 - 1;
      goto LABEL_8;
    }
    sub_28C91C0(v21, v63, v62, a4);
    result = v21 - (char *)a1;
    if ( v21 - (char *)a1 > 256 )
    {
      if ( v62 )
      {
        v63 = v21;
        continue;
      }
      goto LABEL_46;
    }
    return result;
  }
}
