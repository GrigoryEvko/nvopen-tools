// Function: sub_D72D40
// Address: 0xd72d40
//
unsigned __int64 __fastcall sub_D72D40(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 *v9; // rdx
  __int64 v10; // r11
  unsigned __int64 v11; // r14
  int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // ecx
  _QWORD *v21; // rax
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rsi
  char v28; // cl
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // r13
  _QWORD *v36; // rdi
  _QWORD *v37; // rdx
  _QWORD *v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // rbx
  _QWORD *v42; // r12
  unsigned __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 *v47; // rax
  _QWORD *v48; // rcx
  __int64 v49; // rsi
  _QWORD *v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rdx
  _QWORD *v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rdi
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rsi
  _QWORD *v59; // rcx
  __int64 v60; // rax
  int v61; // ebx
  int v62; // edx
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // [rsp+0h] [rbp-180h]
  __int64 v68; // [rsp+18h] [rbp-168h]
  char v69; // [rsp+27h] [rbp-159h]
  __int64 v70; // [rsp+28h] [rbp-158h]
  _QWORD *v71; // [rsp+28h] [rbp-158h]
  __int64 v72; // [rsp+28h] [rbp-158h]
  __int64 v73; // [rsp+30h] [rbp-150h] BYREF
  _QWORD v74[3]; // [rsp+38h] [rbp-148h] BYREF
  __int64 v75; // [rsp+50h] [rbp-130h] BYREF
  __int64 v76; // [rsp+58h] [rbp-128h] BYREF
  __int64 v77; // [rsp+60h] [rbp-120h]
  __int64 v78; // [rsp+68h] [rbp-118h]
  _BYTE *v79; // [rsp+80h] [rbp-100h] BYREF
  __int64 v80; // [rsp+88h] [rbp-F8h]
  _BYTE v81[16]; // [rsp+90h] [rbp-F0h] BYREF
  char v82; // [rsp+A0h] [rbp-E0h]

  v5 = a3;
  v6 = *(unsigned int *)(a3 + 24);
  v7 = *(_QWORD *)(a3 + 8);
  if ( (_DWORD)v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 32LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      if ( v9 != (__int64 *)(v7 + 32 * v6) )
        return v9[3];
    }
    else
    {
      v13 = 1;
      while ( v10 != -4096 )
      {
        v20 = v13 + 1;
        v8 = (v6 - 1) & (v13 + v8);
        v9 = (__int64 *)(v7 + 32LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v13 = v20;
      }
    }
  }
  v14 = *(_QWORD *)(*a1 + 8LL);
  if ( a2 )
  {
    v15 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v16 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v15 = 0;
    v16 = 0;
  }
  if ( v16 >= *(_DWORD *)(v14 + 32) || !*(_QWORD *)(*(_QWORD *)(v14 + 24) + 8 * v15) )
    return *(_QWORD *)(*a1 + 128LL);
  v70 = sub_AA5510(a2);
  v68 = (__int64)(a1 + 51);
  if ( v70 )
  {
    sub_D695C0((__int64)&v79, (__int64)(a1 + 51), (__int64 *)a2, v17, v18, v19);
    v75 = a2;
    v11 = sub_D740C0(a1, v70, v5);
    v76 = 6;
    v77 = 0;
    v78 = 0;
    sub_D67D70(&v76, v11);
    sub_D6C550((__int64)&v79, v5, &v75, &v76);
    sub_D68D70(&v76);
    return v11;
  }
  if ( !*((_BYTE *)a1 + 436) )
  {
    if ( !sub_C8CA60(v68, a2) )
      goto LABEL_25;
LABEL_23:
    v23 = sub_10420D0(*a1, a2);
    v75 = a2;
    v11 = v23;
    v76 = 6;
    v77 = 0;
    v78 = 0;
    sub_D67D70(&v76, v23);
    sub_D6C550((__int64)&v79, v5, &v75, &v76);
    sub_D68D70(&v76);
    return v11;
  }
  v21 = (_QWORD *)a1[52];
  v22 = &v21[*((unsigned int *)a1 + 107)];
  if ( v21 != v22 )
  {
    while ( a2 != *v21 )
    {
      if ( v22 == ++v21 )
        goto LABEL_25;
    }
    goto LABEL_23;
  }
LABEL_25:
  sub_D695C0((__int64)&v79, v68, (__int64 *)a2, v17, v18, v19);
  v69 = v82;
  if ( !v82 )
    BUG();
  v79 = v81;
  v80 = 0x800000000LL;
  v24 = *(_QWORD *)(a2 + 16);
  if ( v24 )
  {
    while ( (unsigned __int8)(**(_BYTE **)(v24 + 24) - 30) > 0xAu )
    {
      v24 = *(_QWORD *)(v24 + 8);
      if ( !v24 )
        goto LABEL_40;
    }
    v73 = v24;
    do
    {
      v33 = *(_QWORD *)(*(_QWORD *)(v24 + 24) + 40LL);
      v34 = *(_QWORD *)(*a1 + 8LL);
      if ( v33 )
      {
        v25 = (unsigned int)(*(_DWORD *)(v33 + 44) + 1);
        v26 = *(_DWORD *)(v33 + 44) + 1;
      }
      else
      {
        v25 = 0;
        v26 = 0;
      }
      if ( v26 < *(_DWORD *)(v34 + 32) && *(_QWORD *)(*(_QWORD *)(v34 + 24) + 8 * v25) )
      {
        v27 = sub_D740C0(a1, v33, v5);
        if ( v70 )
        {
          v28 = v69;
          if ( v27 != v70 )
            v28 = 0;
          v69 = v28;
        }
        else
        {
          v70 = v27;
        }
      }
      else
      {
        v27 = *(_QWORD *)(*a1 + 128LL);
      }
      v75 = 6;
      v76 = 0;
      v77 = 0;
      sub_D67D70(&v75, v27);
      sub_D6B460((__int64)&v79, (char *)&v75, v29, v30, v31, v32);
      sub_D68D70(&v75);
      v73 = *(_QWORD *)(v73 + 8);
      sub_D4B000(&v73);
      v24 = v73;
    }
    while ( v73 );
    v67 = sub_D68B40(*a1, a2);
    v43 = sub_D72B90(a1, v67, (__int64 *)&v79);
    v11 = v43;
    if ( v67 != v43 )
      goto LABEL_41;
    if ( v70 && v69 )
    {
      if ( v43 )
      {
        sub_BD84D0(v43, v70);
        sub_D6E4B0(a1, v11, 0, v44, v45, v46);
      }
      v11 = v70;
      goto LABEL_41;
    }
    if ( v69 != 1 || !v70 )
    {
LABEL_64:
      if ( !v11 )
        v11 = sub_10420D0(*a1, a2);
      if ( (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != 0 )
      {
        v48 = *(_QWORD **)(v11 - 8);
        v49 = (__int64)v79;
        v50 = v48;
        v51 = (__int64)v79;
        v52 = v48;
        while ( *v52 == *(_QWORD *)(v51 + 16) )
        {
          v52 += 4;
          v51 += 24;
          if ( v52 == &v48[4 * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)] )
            goto LABEL_41;
        }
        if ( 24LL * (unsigned int)v80 )
        {
          v53 = &v48[0xAAAAAAAAAAAAAACLL * ((24LL * (unsigned int)v80) >> 3)];
          do
          {
            v54 = *(_QWORD *)(v49 + 16);
            if ( *v50 )
            {
              v55 = v50[1];
              *(_QWORD *)v50[2] = v55;
              if ( v55 )
                *(_QWORD *)(v55 + 16) = v50[2];
            }
            *v50 = v54;
            if ( v54 )
            {
              v56 = *(_QWORD *)(v54 + 16);
              v50[1] = v56;
              if ( v56 )
                *(_QWORD *)(v56 + 16) = v50 + 1;
              v50[2] = v54 + 16;
              *(_QWORD *)(v54 + 16) = v50;
            }
            v50 += 4;
            v49 += 24;
          }
          while ( v50 != v53 );
          v48 = *(_QWORD **)(v11 - 8);
        }
        v71 = &v48[4 * *(unsigned int *)(v11 + 76)];
        v75 = *(_QWORD *)(a2 + 16);
        sub_D4B000(&v75);
        v57 = v75;
        if ( v75 )
        {
          v58 = *(_QWORD *)(v75 + 24);
          v59 = v71;
LABEL_84:
          *v59++ = *(_QWORD *)(v58 + 40);
          while ( 1 )
          {
            v57 = *(_QWORD *)(v57 + 8);
            if ( !v57 )
              break;
            v58 = *(_QWORD *)(v57 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v58 - 30) <= 0xAu )
              goto LABEL_84;
          }
        }
      }
      else
      {
        v60 = *(_QWORD *)(a2 + 16);
        if ( v60 )
        {
          while ( (unsigned __int8)(**(_BYTE **)(v60 + 24) - 30) > 0xAu )
          {
            v60 = *(_QWORD *)(v60 + 8);
            if ( !v60 )
              goto LABEL_93;
          }
          v75 = v60;
          v72 = v5;
          v61 = 0;
          do
          {
            v62 = v61++;
            sub_D689D0(v11, *(_QWORD *)&v79[24 * v62 + 16], *(_QWORD *)(*(_QWORD *)(v60 + 24) + 40LL));
            v75 = *(_QWORD *)(v75 + 8);
            sub_D4B000(&v75);
            v60 = v75;
          }
          while ( v75 );
          v5 = v72;
        }
LABEL_93:
        sub_D68D20((__int64)&v75, 2u, v11);
        sub_D6B260((__int64)(a1 + 1), (char *)&v75, v63, v64, v65, v66);
        sub_D68D70(&v75);
      }
    }
  }
  else
  {
LABEL_40:
    v35 = sub_D68B40(*a1, a2);
    v11 = sub_D72B90(a1, v35, (__int64 *)&v79);
    if ( v11 == v35 )
      goto LABEL_64;
  }
LABEL_41:
  if ( *((_BYTE *)a1 + 436) )
  {
    v36 = (_QWORD *)a1[52];
    v37 = &v36[*((unsigned int *)a1 + 107)];
    v38 = v36;
    if ( v36 != v37 )
    {
      while ( a2 != *v38 )
      {
        if ( v37 == ++v38 )
          goto LABEL_47;
      }
      v39 = (unsigned int)(*((_DWORD *)a1 + 107) - 1);
      *((_DWORD *)a1 + 107) = v39;
      *v38 = v36[v39];
      ++a1[51];
    }
  }
  else
  {
    v47 = sub_C8CA60(v68, a2);
    if ( v47 )
    {
      *v47 = -2;
      ++*((_DWORD *)a1 + 108);
      ++a1[51];
    }
  }
LABEL_47:
  v73 = a2;
  v74[0] = 6;
  v74[1] = 0;
  v74[2] = 0;
  sub_D67D70(v74, v11);
  v40 = v5;
  sub_D6C550((__int64)&v75, v5, &v73, v74);
  sub_D68D70(v74);
  v41 = (__int64)v79;
  v42 = &v79[24 * (unsigned int)v80];
  if ( v79 != (_BYTE *)v42 )
  {
    do
    {
      v42 -= 3;
      sub_D68D70(v42);
    }
    while ( (_QWORD *)v41 != v42 );
    v42 = v79;
  }
  if ( v42 != (_QWORD *)v81 )
    _libc_free(v42, v40);
  return v11;
}
