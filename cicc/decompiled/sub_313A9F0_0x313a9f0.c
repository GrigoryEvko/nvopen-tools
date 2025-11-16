// Function: sub_313A9F0
// Address: 0x313a9f0
//
__int64 __fastcall sub_313A9F0(__int64 a1, __int64 *a2, unsigned int a3, int a4, unsigned int a5)
{
  __int64 v5; // r9
  __int64 v7; // r14
  __int64 v8; // r13
  unsigned __int64 v10; // rdx
  __int64 v12; // rsi
  __int64 v13; // r11
  __int64 *v14; // rcx
  unsigned int i; // eax
  _QWORD *v16; // rdi
  __int64 *v17; // r8
  unsigned int v18; // eax
  unsigned __int64 *v19; // r10
  unsigned __int64 v20; // rdi
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 v25; // rdi
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 *v28; // rax
  __int64 **v29; // rdi
  __int64 v30; // rax
  unsigned __int64 *v31; // r10
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // r13
  int v36; // eax
  int v37; // edx
  __int64 v38; // rdi
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 *v41; // r10
  __int64 v42; // r12
  int v43; // [rsp+8h] [rbp-A8h]
  unsigned __int64 *v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  unsigned __int64 *v46; // [rsp+8h] [rbp-A8h]
  __int64 *v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+18h] [rbp-98h]
  __int64 *v49[6]; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v50; // [rsp+50h] [rbp-60h] BYREF
  __int64 v51; // [rsp+58h] [rbp-58h]
  __int16 v52; // [rsp+70h] [rbp-40h]

  v5 = a1 + 680;
  v7 = a5;
  v8 = a4 | 2;
  v10 = a5 | (unsigned __int64)(v8 << 31);
  v50 = a2;
  v12 = *(unsigned int *)(a1 + 704);
  v51 = v10;
  if ( !(_DWORD)v12 )
  {
    ++*(_QWORD *)(a1 + 680);
    v49[0] = 0;
    goto LABEL_36;
  }
  v13 = *(_QWORD *)(a1 + 688);
  v43 = 1;
  v14 = 0;
  for ( i = (v12 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL * v10) >> 31) ^ (484763065 * (_DWORD)v10)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v10) >> 31) ^ (484763065 * v10)))); ; i = (v12 - 1) & v18 )
  {
    v16 = (_QWORD *)(v13 + 24LL * i);
    v17 = (__int64 *)*v16;
    if ( a2 == (__int64 *)*v16 && v10 == v16[1] )
    {
      v19 = v16 + 2;
      goto LABEL_12;
    }
    if ( v17 == (__int64 *)-4096LL )
      break;
    if ( v17 == (__int64 *)-8192LL && v16[1] == -2 && !v14 )
      v14 = (__int64 *)(v13 + 24LL * i);
LABEL_9:
    v18 = v43 + i;
    ++v43;
  }
  if ( v16[1] != -1 )
    goto LABEL_9;
  v36 = *(_DWORD *)(a1 + 696);
  if ( !v14 )
    v14 = v16;
  ++*(_QWORD *)(a1 + 680);
  v37 = v36 + 1;
  v49[0] = v14;
  if ( 4 * (v36 + 1) < (unsigned int)(3 * v12) )
  {
    v38 = (__int64)a2;
    if ( (int)v12 - *(_DWORD *)(a1 + 700) - v37 > (unsigned int)v12 >> 3 )
      goto LABEL_30;
    goto LABEL_37;
  }
LABEL_36:
  LODWORD(v12) = 2 * v12;
LABEL_37:
  v45 = v5;
  sub_313A720(v5, v12);
  v12 = (__int64)&v50;
  sub_3139BD0(v45, (__int64 *)&v50, v49);
  v38 = (__int64)v50;
  v14 = v49[0];
  v37 = *(_DWORD *)(a1 + 696) + 1;
LABEL_30:
  *(_DWORD *)(a1 + 696) = v37;
  if ( *v14 != -4096 || v14[1] != -1 )
    --*(_DWORD *)(a1 + 700);
  *v14 = v38;
  v39 = v51;
  v19 = (unsigned __int64 *)(v14 + 2);
  v14[2] = 0;
  v14[1] = v39;
LABEL_12:
  v20 = *v19;
  if ( !*v19 )
  {
    v44 = v19;
    v22 = (__int64 *)sub_AD6530(*(_QWORD *)(a1 + 2632), v12);
    v23 = *(_QWORD *)(a1 + 2632);
    v49[0] = v22;
    v24 = (__int64 *)sub_AD64C0(v23, (unsigned int)v8, 0);
    v25 = *(_QWORD *)(a1 + 2632);
    v49[1] = v24;
    v26 = (__int64 *)sub_AD64C0(v25, v7, 0);
    v27 = *(_QWORD *)(a1 + 2632);
    v49[2] = v26;
    v28 = (__int64 *)sub_AD64C0(v27, a3, 0);
    v29 = *(__int64 ***)(a1 + 2776);
    v49[4] = a2;
    v49[3] = v28;
    v30 = sub_AD24A0(v29, (__int64 *)v49, 5);
    v31 = v44;
    v32 = v30;
    v33 = *(_QWORD *)(a1 + 504);
    v34 = *(_QWORD *)(v33 + 16);
    v35 = v33 + 8;
    if ( v34 != v33 + 8 )
    {
      do
      {
        if ( !v34 )
          BUG();
        if ( *(_QWORD *)(a1 + 2776) == *(_QWORD *)(v34 - 32) && !sub_B2FC80(v34 - 56) && v32 == *(_QWORD *)(v34 - 88) )
          *v44 = v34 - 56;
        v34 = *(_QWORD *)(v34 + 8);
      }
      while ( v35 != v34 );
      v31 = v44;
    }
    if ( !*v31 )
    {
      v46 = v31;
      v52 = 257;
      BYTE4(v48) = 1;
      LODWORD(v48) = *(_DWORD *)(*(_QWORD *)(a1 + 504) + 324LL);
      v40 = sub_BD2C40(88, unk_3F0FAE8);
      v41 = (__int64 *)v46;
      v42 = (__int64)v40;
      if ( v40 )
      {
        sub_B30000((__int64)v40, *(_QWORD *)(a1 + 504), *(_QWORD **)(a1 + 2776), 1, 8, v32, (__int64)&v50, 0, 0, v48, 0);
        v41 = (__int64 *)v46;
      }
      v47 = v41;
      *(_BYTE *)(v42 + 32) = *(_BYTE *)(v42 + 32) & 0x3F | 0x80;
      sub_B2F770(v42, 3u);
      v31 = (unsigned __int64 *)v47;
      *v47 = v42;
    }
    v20 = *v31;
  }
  return sub_ADB060(v20, *(_QWORD *)(a1 + 2784));
}
