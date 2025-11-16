// Function: sub_1BAEFD0
// Address: 0x1baefd0
//
__int64 __fastcall sub_1BAEFD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // r12
  int v5; // eax
  __int64 v6; // rdi
  int v7; // r8d
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 *v12; // rsi
  __int64 *v13; // rdx
  __int64 *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r15
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned int v21; // eax
  __int64 *v22; // rcx
  __int64 v23; // rsi
  unsigned __int8 v24; // dl
  __int64 v25; // r14
  _QWORD *v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  _BYTE *v29; // rdi
  __int64 v30; // r15
  int v31; // eax
  int v32; // r8d
  int v33; // r9d
  _QWORD *v34; // rsi
  _QWORD *v35; // rax
  _QWORD *v36; // r14
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 *v39; // rsi
  __int64 *v40; // rdx
  int v41; // ecx
  int v42; // r9d
  _QWORD *v44; // rdx
  int v45; // eax
  int v46; // r9d
  char v47; // [rsp+1Fh] [rbp-C1h]
  __int64 v48; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+28h] [rbp-B8h]
  __int64 v50; // [rsp+30h] [rbp-B0h]
  __int64 v51; // [rsp+38h] [rbp-A8h]
  __int64 v52; // [rsp+40h] [rbp-A0h]
  __int64 v53; // [rsp+48h] [rbp-98h]
  __int64 v54; // [rsp+50h] [rbp-90h]
  _BYTE *v55; // [rsp+60h] [rbp-80h] BYREF
  __int64 v56; // [rsp+68h] [rbp-78h]
  _BYTE v57[112]; // [rsp+70h] [rbp-70h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_DWORD *)(v3 + 24);
  if ( v5 )
  {
    v6 = *(_QWORD *)(v3 + 8);
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
    {
LABEL_3:
      v2 = v9[1];
    }
    else
    {
      v45 = 1;
      while ( v10 != -8 )
      {
        v46 = v45 + 1;
        v8 = v7 & (v45 + v8);
        v9 = (__int64 *)(v6 + 16LL * v8);
        v10 = *v9;
        if ( v4 == *v9 )
          goto LABEL_3;
        v45 = v46;
      }
      v2 = 0;
    }
  }
  v11 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v12 = *(__int64 **)(a2 - 8);
    v13 = &v12[v11];
  }
  else
  {
    v13 = (__int64 *)a2;
    v12 = (__int64 *)(a2 - v11 * 8);
  }
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  sub_1BAEC70((__int64)&v48, v12, v13);
  v14 = (__int64 *)v57;
  v55 = v57;
  v56 = 0x800000000LL;
  v15 = 0;
  while ( 2 )
  {
    sub_1BAEE20((__int64)&v48, v14, &v14[v15]);
    LODWORD(v56) = 0;
    v16 = v53;
    if ( v53 == v52 )
    {
      v29 = v55;
      break;
    }
    v47 = 0;
    while ( 2 )
    {
      v20 = *(_QWORD *)(v16 - 8);
      if ( (_DWORD)v51 )
      {
        v21 = (v51 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v22 = (__int64 *)(v49 + 8LL * v21);
        v23 = *v22;
        if ( v20 == *v22 )
        {
LABEL_18:
          *v22 = -16;
          LODWORD(v50) = v50 - 1;
          ++HIDWORD(v50);
        }
        else
        {
          v41 = 1;
          while ( v23 != -8 )
          {
            v42 = v41 + 1;
            v21 = (v51 - 1) & (v41 + v21);
            v22 = (__int64 *)(v49 + 8LL * v21);
            v23 = *v22;
            if ( v20 == *v22 )
              goto LABEL_18;
            v41 = v42;
          }
        }
      }
      v16 = v53 - 8;
      v53 -= 8;
      v24 = *(_BYTE *)(v20 + 16);
      if ( v24 <= 0x17u )
        goto LABEL_15;
      if ( v24 == 77 )
        goto LABEL_15;
      v25 = *(_QWORD *)(v20 + 40);
      if ( v4 == v25 )
        goto LABEL_15;
      v26 = *(_QWORD **)(v2 + 72);
      v18 = *(_QWORD **)(v2 + 64);
      if ( v26 == v18 )
      {
        v17 = &v18[*(unsigned int *)(v2 + 84)];
        if ( v18 == v17 )
        {
          v44 = *(_QWORD **)(v2 + 64);
        }
        else
        {
          do
          {
            if ( v25 == *v18 )
              break;
            ++v18;
          }
          while ( v17 != v18 );
          v44 = v17;
        }
      }
      else
      {
        v17 = &v26[*(unsigned int *)(v2 + 80)];
        v18 = sub_16CC9F0(v2 + 56, *(_QWORD *)(v20 + 40));
        if ( v25 == *v18 )
        {
          v27 = *(_QWORD *)(v2 + 72);
          if ( v27 == *(_QWORD *)(v2 + 64) )
            v28 = *(unsigned int *)(v2 + 84);
          else
            v28 = *(unsigned int *)(v2 + 80);
          v44 = (_QWORD *)(v27 + 8 * v28);
        }
        else
        {
          v19 = *(_QWORD *)(v2 + 72);
          if ( v19 != *(_QWORD *)(v2 + 64) )
          {
            v18 = (_QWORD *)(v19 + 8LL * *(unsigned int *)(v2 + 80));
            goto LABEL_12;
          }
          v18 = (_QWORD *)(v19 + 8LL * *(unsigned int *)(v2 + 84));
          v44 = v18;
        }
      }
      while ( v44 != v18 && *v18 >= 0xFFFFFFFFFFFFFFFELL )
        ++v18;
LABEL_12:
      if ( v18 == v17 || (unsigned __int8)sub_15F3040(v20) || sub_15F3330(v20) )
        goto LABEL_14;
      v30 = *(_QWORD *)(v20 + 8);
      if ( !v30 )
      {
LABEL_50:
        v38 = sub_157EE30(v4);
        if ( v38 )
          v38 -= 24;
        sub_15F22F0((_QWORD *)v20, v38);
        if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
        {
          v39 = *(__int64 **)(v20 - 8);
          v40 = &v39[3 * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF)];
        }
        else
        {
          v40 = (__int64 *)v20;
          v39 = (__int64 *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
        }
        sub_1BAEC70((__int64)&v48, v39, v40);
        v47 = 1;
        v16 = v53;
        goto LABEL_15;
      }
      while ( 1 )
      {
        v35 = sub_1648700(v30);
        v36 = v35;
        if ( *((_BYTE *)v35 + 16) == 77 )
          break;
        if ( v4 != v35[5] )
          goto LABEL_46;
LABEL_43:
        v30 = *(_QWORD *)(v30 + 8);
        if ( !v30 )
          goto LABEL_50;
      }
      v31 = sub_1648720(v30);
      if ( (*((_BYTE *)v36 + 23) & 0x40) != 0 )
        v34 = (_QWORD *)*(v36 - 1);
      else
        v34 = &v36[-3 * (*((_DWORD *)v36 + 5) & 0xFFFFFFF)];
      if ( v4 == v34[3 * *((unsigned int *)v36 + 14) + 1 + v31] )
        goto LABEL_43;
LABEL_46:
      v37 = (unsigned int)v56;
      if ( (unsigned int)v56 >= HIDWORD(v56) )
      {
        sub_16CD150((__int64)&v55, v57, 0, 8, v32, v33);
        v37 = (unsigned int)v56;
      }
      *(_QWORD *)&v55[8 * v37] = v20;
      LODWORD(v56) = v56 + 1;
LABEL_14:
      v16 = v53;
LABEL_15:
      if ( v52 != v16 )
        continue;
      break;
    }
    v14 = (__int64 *)v55;
    v29 = v55;
    if ( v47 )
    {
      v15 = (unsigned int)v56;
      continue;
    }
    break;
  }
  if ( v29 != v57 )
    _libc_free((unsigned __int64)v29);
  if ( v52 )
    j_j___libc_free_0(v52, v54 - v52);
  return j___libc_free_0(v49);
}
