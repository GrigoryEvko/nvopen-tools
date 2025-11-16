// Function: sub_1530240
// Address: 0x1530240
//
void __fastcall sub_1530240(__int64 ***a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r10
  __int64 v8; // r14
  int v9; // ecx
  __int64 v10; // rax
  int v11; // ecx
  __int64 **v12; // r10
  __int64 *v13; // rdi
  unsigned int v14; // esi
  __int64 **v15; // rdx
  __int64 *v16; // r14
  __int64 **v17; // r15
  int v18; // ecx
  int v19; // r13d
  unsigned int v20; // ecx
  int v21; // r13d
  __int64 *v22; // rax
  __int64 v23; // rdx
  int v24; // eax
  int v25; // ebx
  unsigned int v26; // r9d
  int v27; // r13d
  unsigned int v28; // ebx
  unsigned int v29; // ecx
  unsigned __int64 v30; // r14
  unsigned int v31; // eax
  __int64 **v32; // r13
  unsigned __int64 v33; // r15
  unsigned int v34; // r14d
  int v35; // eax
  int v36; // eax
  __int64 *v37; // rax
  __int64 v38; // rdx
  unsigned int v39; // r14d
  int v40; // edx
  unsigned int v41; // ecx
  int v42; // ebx
  __int64 *v43; // rdi
  __int64 v44; // rdx
  int v45; // edx
  unsigned int v46; // eax
  __int64 **v47; // r13
  unsigned int j; // r15d
  int v49; // edx
  int v50; // edx
  __int64 *v51; // rdi
  __int64 v52; // rdx
  int v53; // edx
  int v54; // ebx
  __int64 *v55; // r14
  __int64 v56; // rdx
  int v57; // edx
  int v58; // r8d
  __int64 v59; // [rsp+10h] [rbp-2E0h]
  __int64 v60; // [rsp+20h] [rbp-2D0h]
  __int64 i; // [rsp+48h] [rbp-2A8h]
  int v63; // [rsp+50h] [rbp-2A0h]
  int v64; // [rsp+54h] [rbp-29Ch]
  int v65; // [rsp+54h] [rbp-29Ch]
  unsigned int v66; // [rsp+54h] [rbp-29Ch]
  int v67; // [rsp+54h] [rbp-29Ch]
  __int64 *v68; // [rsp+58h] [rbp-298h]
  __int64 *v69; // [rsp+58h] [rbp-298h]
  __int64 *v70; // [rsp+58h] [rbp-298h]
  __int64 v71; // [rsp+58h] [rbp-298h]
  unsigned int v72; // [rsp+58h] [rbp-298h]
  _BYTE *v73; // [rsp+60h] [rbp-290h] BYREF
  __int64 v74; // [rsp+68h] [rbp-288h]
  _BYTE v75[64]; // [rsp+70h] [rbp-280h] BYREF
  _BYTE *v76; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v77; // [rsp+B8h] [rbp-238h]
  _BYTE v78[560]; // [rsp+C0h] [rbp-230h] BYREF

  sub_1526BE0(*a1, 0x10u, 3u);
  v76 = v78;
  v77 = 0x4000000000LL;
  if ( (*(_BYTE *)(a2 + 34) & 0x10) != 0 )
  {
    sub_1524A90((__int64)a1, (__int64)&v76, a2);
    sub_152F3D0(*a1, 0xBu, (__int64)&v76, 0);
    LODWORD(v77) = 0;
  }
  v73 = v75;
  v74 = 0x400000000LL;
  v59 = a2 + 72;
  v60 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != v60 )
  {
    do
    {
      if ( !v60 )
        BUG();
      for ( i = *(_QWORD *)(v60 + 24); v60 + 16 != i; i = *(_QWORD *)(i + 8) )
      {
        LODWORD(v74) = 0;
        if ( !i )
          BUG();
        if ( *(__int16 *)(i - 6) < 0 )
        {
          sub_161F980(i - 24, &v73);
          if ( (_DWORD)v74 )
          {
            v2 = (unsigned int)sub_153E760(a1 + 3, i - 24);
            v3 = (unsigned int)v77;
            if ( (unsigned int)v77 >= HIDWORD(v77) )
            {
              sub_16CD150(&v76, v78, 0, 8);
              v3 = (unsigned int)v77;
            }
            *(_QWORD *)&v76[8 * v3] = v2;
            v4 = (unsigned int)(v77 + 1);
            LODWORD(v77) = v77 + 1;
            if ( (_DWORD)v74 )
            {
              v5 = 0;
              v6 = 16LL * (unsigned int)v74;
              do
              {
                v7 = *(unsigned int *)&v73[v5];
                if ( HIDWORD(v77) <= (unsigned int)v4 )
                {
                  v71 = *(unsigned int *)&v73[v5];
                  sub_16CD150(&v76, v78, 0, 8);
                  v4 = (unsigned int)v77;
                  v7 = v71;
                }
                *(_QWORD *)&v76[8 * v4] = v7;
                v8 = 0xFFFFFFFFLL;
                v9 = *((_DWORD *)a1 + 76);
                v10 = (unsigned int)(v77 + 1);
                LODWORD(v77) = v77 + 1;
                if ( v9 )
                {
                  v11 = v9 - 1;
                  v12 = a1[36];
                  v13 = *(__int64 **)&v73[v5 + 8];
                  v14 = v11 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
                  v15 = &v12[2 * v14];
                  v16 = *v15;
                  if ( v13 == *v15 )
                  {
LABEL_19:
                    v8 = (unsigned int)(*((_DWORD *)v15 + 3) - 1);
                  }
                  else
                  {
                    v57 = 1;
                    while ( v16 != (__int64 *)-4LL )
                    {
                      v58 = v57 + 1;
                      v14 = v11 & (v57 + v14);
                      v15 = &v12[2 * v14];
                      v16 = *v15;
                      if ( v13 == *v15 )
                        goto LABEL_19;
                      v57 = v58;
                    }
                    v8 = 0xFFFFFFFFLL;
                  }
                }
                if ( HIDWORD(v77) <= (unsigned int)v10 )
                {
                  sub_16CD150(&v76, v78, 0, 8);
                  v10 = (unsigned int)v77;
                }
                v5 += 16;
                *(_QWORD *)&v76[8 * v10] = v8;
                v4 = (unsigned int)(v77 + 1);
                LODWORD(v77) = v77 + 1;
              }
              while ( v6 != v5 );
            }
            v17 = *a1;
            sub_1524D80(*a1, 3u, *((_DWORD *)*a1 + 4));
            sub_1524D80(v17, 0xBu, 6);
            if ( (unsigned int)v4 > 0x1F )
            {
              v64 = v4;
              do
              {
                while ( 1 )
                {
                  v18 = *((_DWORD *)v17 + 2);
                  v19 = (v4 & 0x1F | 0x20) << v18;
                  v20 = v18 + 6;
                  v21 = *((_DWORD *)v17 + 3) | v19;
                  *((_DWORD *)v17 + 3) = v21;
                  if ( v20 > 0x1F )
                    break;
                  LODWORD(v4) = (unsigned int)v4 >> 5;
                  *((_DWORD *)v17 + 2) = v20;
                  if ( (unsigned int)v4 <= 0x1F )
                    goto LABEL_32;
                }
                v22 = *v17;
                v23 = *((unsigned int *)*v17 + 2);
                if ( (unsigned __int64)*((unsigned int *)*v17 + 3) - v23 <= 3 )
                {
                  v69 = *v17;
                  sub_16CD150(*v17, v22 + 2, v23 + 4, 1);
                  v22 = v69;
                  v23 = *((unsigned int *)v69 + 2);
                }
                *(_DWORD *)(*v22 + v23) = v21;
                *((_DWORD *)v22 + 2) += 4;
                v24 = *((_DWORD *)v17 + 2);
                v25 = (unsigned __int64)(v4 & 0x1F | 0x20) >> (32 - (unsigned __int8)v24);
                if ( !v24 )
                  v25 = 0;
                LODWORD(v4) = (unsigned int)v4 >> 5;
                *((_DWORD *)v17 + 3) = v25;
                *((_DWORD *)v17 + 2) = ((_BYTE)v24 + 6) & 0x1F;
              }
              while ( (unsigned int)v4 > 0x1F );
LABEL_32:
              v26 = v4;
              LODWORD(v4) = v64;
              sub_1524D80(v17, v26, 6);
LABEL_33:
              v63 = v4;
              v27 = 0;
LABEL_34:
              v28 = *((_DWORD *)v17 + 3);
              v29 = *((_DWORD *)v17 + 2);
              v30 = *(_QWORD *)&v76[8 * v27];
              v31 = v30;
              if ( v30 == (unsigned int)v30 )
              {
                if ( (unsigned int)v30 > 0x1F )
                {
                  v67 = v27;
                  v47 = v17;
                  for ( j = v30; j > 0x1F; j >>= 5 )
                  {
                    v50 = (j & 0x1F | 0x20) << v29;
                    v29 += 6;
                    v28 |= v50;
                    *((_DWORD *)v47 + 3) = v28;
                    if ( v29 > 0x1F )
                    {
                      v51 = *v47;
                      v52 = *((unsigned int *)*v47 + 2);
                      if ( (unsigned __int64)*((unsigned int *)*v47 + 3) - v52 <= 3 )
                      {
                        v70 = *v47;
                        sub_16CD150(v51, v51 + 2, v52 + 4, 1);
                        v51 = v70;
                        v52 = *((unsigned int *)v70 + 2);
                      }
                      *(_DWORD *)(*v51 + v52) = v28;
                      v28 = 0;
                      *((_DWORD *)v51 + 2) += 4;
                      v49 = *((_DWORD *)v47 + 2);
                      if ( v49 )
                        v28 = (j & 0x1F | 0x20) >> (32 - v49);
                      v29 = ((_BYTE)v49 + 6) & 0x1F;
                      *((_DWORD *)v47 + 3) = v28;
                    }
                    *((_DWORD *)v47 + 2) = v29;
                  }
                  v31 = j;
                  v17 = v47;
                  v27 = v67;
                }
                v53 = v31 << v29;
                v41 = v29 + 6;
                v54 = v53 | v28;
                *((_DWORD *)v17 + 3) = v54;
                if ( v41 <= 0x1F )
                  goto LABEL_54;
                v55 = *v17;
                v56 = *((unsigned int *)*v17 + 2);
                if ( (unsigned __int64)*((unsigned int *)*v17 + 3) - v56 <= 3 )
                {
                  v72 = v31;
                  sub_16CD150(*v17, v55 + 2, v56 + 4, 1);
                  v56 = *((unsigned int *)v55 + 2);
                  v31 = v72;
                }
                *(_DWORD *)(*v55 + v56) = v54;
                *((_DWORD *)v55 + 2) += 4;
LABEL_49:
                v45 = *((_DWORD *)v17 + 2);
                v46 = v31 >> (32 - v45);
                if ( !v45 )
                  v46 = 0;
                *((_DWORD *)v17 + 3) = v46;
                *((_DWORD *)v17 + 2) = ((_BYTE)v45 + 6) & 0x1F;
              }
              else
              {
                if ( v30 > 0x1F )
                {
                  v65 = v27;
                  v32 = v17;
                  v33 = v30;
                  v34 = v28;
                  do
                  {
                    v36 = (v33 & 0x1F | 0x20) << v29;
                    v29 += 6;
                    v34 |= v36;
                    *((_DWORD *)v32 + 3) = v34;
                    if ( v29 > 0x1F )
                    {
                      v37 = *v32;
                      v38 = *((unsigned int *)*v32 + 2);
                      if ( (unsigned __int64)*((unsigned int *)*v32 + 3) - v38 <= 3 )
                      {
                        v68 = *v32;
                        sub_16CD150(*v32, v37 + 2, v38 + 4, 1);
                        v37 = v68;
                        v38 = *((unsigned int *)v68 + 2);
                      }
                      *(_DWORD *)(*v37 + v38) = v34;
                      v34 = 0;
                      *((_DWORD *)v37 + 2) += 4;
                      v35 = *((_DWORD *)v32 + 2);
                      if ( v35 )
                        v34 = (v33 & 0x1F | 0x20) >> (32 - (unsigned __int8)v35);
                      v29 = ((_BYTE)v35 + 6) & 0x1F;
                      *((_DWORD *)v32 + 3) = v34;
                    }
                    v33 >>= 5;
                    *((_DWORD *)v32 + 2) = v29;
                  }
                  while ( v33 > 0x1F );
                  v28 = v34;
                  v39 = v33;
                  v17 = v32;
                  v27 = v65;
                  v31 = v39;
                }
                v40 = v31 << v29;
                v41 = v29 + 6;
                v42 = v40 | v28;
                *((_DWORD *)v17 + 3) = v42;
                if ( v41 > 0x1F )
                {
                  v43 = *v17;
                  v44 = *((unsigned int *)*v17 + 2);
                  if ( (unsigned __int64)*((unsigned int *)*v17 + 3) - v44 <= 3 )
                  {
                    v66 = v31;
                    sub_16CD150(v43, v43 + 2, v44 + 4, 1);
                    v31 = v66;
                    v44 = *((unsigned int *)v43 + 2);
                  }
                  *(_DWORD *)(*v43 + v44) = v42;
                  *((_DWORD *)v43 + 2) += 4;
                  goto LABEL_49;
                }
LABEL_54:
                *((_DWORD *)v17 + 2) = v41;
              }
              if ( ++v27 == v63 )
              {
                LODWORD(v77) = 0;
                continue;
              }
              goto LABEL_34;
            }
            sub_1524D80(v17, v4, 6);
            if ( (_DWORD)v4 )
              goto LABEL_33;
            LODWORD(v77) = 0;
          }
        }
      }
      v60 = *(_QWORD *)(v60 + 8);
    }
    while ( v59 != v60 );
  }
  sub_15263C0(*a1);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
}
