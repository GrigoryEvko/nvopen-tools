// Function: sub_2F77D30
// Address: 0x2f77d30
//
__int64 __fastcall sub_2F77D30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int128 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned int *v13; // r12
  unsigned int v14; // r13d
  __int64 v15; // rdi
  __int64 v16; // r10
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 result; // rax
  __int16 v31; // dx
  unsigned int v32; // r14d
  __int64 v33; // rsi
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // [rsp-10h] [rbp-A0h]
  unsigned __int64 v40; // [rsp+0h] [rbp-90h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  __int64 v45; // [rsp+30h] [rbp-60h]
  __int64 v46; // [rsp+30h] [rbp-60h]
  __int64 v47; // [rsp+38h] [rbp-58h]

  if ( !sub_2F753A0(a1) )
    sub_2F75570(a1, a2, v3, v4, v5, v6);
  v40 = 0;
  if ( *(_BYTE *)(a1 + 56) )
    v40 = sub_2F75400((_QWORD *)a1);
  if ( sub_2F753D0(a1) )
  {
    v11 = *(_QWORD *)(a1 + 48);
    if ( *(_BYTE *)(a1 + 56) )
      sub_2F751E0(v11, v40);
    else
      sub_2F75220(v11, *(_QWORD *)(a1 + 64));
  }
  v12 = a1 + 96;
  v13 = *(unsigned int **)a2;
  v47 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v47 )
  {
    do
    {
      v14 = *v13;
      v12 = *v13;
      if ( (*v13 & 0x80000000) != 0 )
        v12 = *(_DWORD *)(a1 + 320) + ((unsigned int)v12 & 0x7FFFFFFF);
      *((_QWORD *)&v7 + 1) = *(_QWORD *)(a1 + 304);
      v8 = *(unsigned int *)(a1 + 104);
      LODWORD(v7) = *(unsigned __int8 *)(*((_QWORD *)&v7 + 1) + (unsigned int)v12);
      if ( (unsigned int)v7 >= (unsigned int)v8 )
        goto LABEL_33;
      v15 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        *((_QWORD *)&v7 + 1) = v15 + 24LL * (unsigned int)v7;
        if ( (_DWORD)v12 == **((_DWORD **)&v7 + 1) )
          break;
        LODWORD(v7) = v7 + 256;
        if ( (unsigned int)v8 <= (unsigned int)v7 )
          goto LABEL_33;
      }
      if ( *((_QWORD *)&v7 + 1) == v15 + 24 * v8 )
      {
LABEL_33:
        v19 = -1;
        v18 = -1;
        v17 = 0;
        v16 = 0;
      }
      else
      {
        v16 = *(_QWORD *)(*((_QWORD *)&v7 + 1) + 8LL);
        v17 = *(_QWORD *)(*((_QWORD *)&v7 + 1) + 16LL);
        v18 = ~v16;
        v19 = ~v17;
      }
      v9 = *((_QWORD *)v13 + 1);
      v10 = *((_QWORD *)v13 + 2);
      *(_QWORD *)&v7 = v9 & v18;
      v20 = v10 & v19;
      if ( v20 | (unsigned __int64)v7 )
      {
        v44 = v7;
        v42 = *((_QWORD *)v13 + 1);
        v43 = *((_QWORD *)v13 + 2);
        v45 = v16;
        sub_2F77040(a1, v12, *((__int64 *)&v7 + 1), v8, v9, v10, v14, v7, v20);
        v12 = v14;
        sub_2F74DB0(a1, v14, v45, v17, v45 | v42, v17 | v43);
        sub_2F74C60(a1 + 96, v14, v21, v22, v23, v24, v14, v44, v20);
        v16 = v45;
      }
      if ( *(_BYTE *)(a1 + 56) )
      {
        v12 = v14;
        v32 = v14;
        v46 = v16;
        *(_QWORD *)&v7 = sub_2F77D00(a1, v14, v40);
        v10 = *((_QWORD *)&v7 + 1);
        v9 = v7;
        if ( v7 != 0 )
        {
          if ( (v14 & 0x80000000) != 0 )
            v14 = *(_DWORD *)(a1 + 320) + (v14 & 0x7FFFFFFF);
          v33 = *(unsigned int *)(a1 + 104);
          v34 = ~(_QWORD)v7;
          v35 = ~*((_QWORD *)&v7 + 1);
          v36 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + v14);
          if ( v36 < (unsigned int)v33 )
          {
            v37 = *(_QWORD *)(a1 + 96);
            while ( 1 )
            {
              v38 = v37 + 24LL * v36;
              if ( v14 == *(_DWORD *)v38 )
                break;
              v36 += 256;
              if ( (unsigned int)v33 <= v36 )
                goto LABEL_43;
            }
            if ( v38 != v37 + 24 * v33 )
            {
              *(_QWORD *)(v38 + 8) &= v34;
              *(_QWORD *)(v38 + 16) &= v35;
            }
          }
LABEL_43:
          v12 = v32;
          sub_2F74F40(a1, v32, v46, v17, v46 & v34, v17 & v35);
        }
      }
      v13 += 6;
    }
    while ( (unsigned int *)v47 != v13 );
  }
  v25 = *(_QWORD *)(a2 + 208);
  v26 = v25 + 24LL * *(unsigned int *)(a2 + 216);
  while ( v26 != v25 )
  {
    v39 = *(_QWORD *)(v25 + 16);
    v25 += 24;
    v27 = sub_2F74C60(
            a1 + 96,
            v12,
            *((__int64 *)&v7 + 1),
            v8,
            v9,
            v10,
            *(_QWORD *)(v25 - 24),
            *(_QWORD *)(v25 - 16),
            v39);
    v12 = *(unsigned int *)(v25 - 24);
    sub_2F74DB0(a1, v12, v27, v28, v27 | *(_QWORD *)(v25 - 16), v28 | *(_QWORD *)(v25 - 8));
  }
  sub_2F77060(a1, *(unsigned int **)(a2 + 416), *(unsigned int *)(a2 + 424));
  v29 = *(_QWORD *)(a1 + 40) + 48LL;
  result = *(_QWORD *)(a1 + 64);
  if ( !result )
    BUG();
  if ( (*(_BYTE *)result & 4) == 0 && (*(_BYTE *)(result + 44) & 8) != 0 )
  {
    do
      result = *(_QWORD *)(result + 8);
    while ( (*(_BYTE *)(result + 44) & 8) != 0 );
  }
LABEL_25:
  for ( result = *(_QWORD *)(result + 8); v29 != result; result = *(_QWORD *)(result + 8) )
  {
    v31 = *(_WORD *)(result + 68);
    if ( (unsigned __int16)(v31 - 14) > 4u && v31 != 24 )
      break;
    if ( (*(_BYTE *)result & 4) != 0 || (*(_BYTE *)(result + 44) & 8) == 0 )
      goto LABEL_25;
    do
      result = *(_QWORD *)(result + 8);
    while ( (*(_BYTE *)(result + 44) & 8) != 0 );
  }
  *(_QWORD *)(a1 + 64) = result;
  return result;
}
