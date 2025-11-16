// Function: sub_1DBCD10
// Address: 0x1dbcd10
//
__int64 __fastcall sub_1DBCD10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned int v4; // r8d
  __int64 *v6; // r13
  __int64 v8; // rax
  __int64 v9; // r11
  unsigned int *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 *v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 *v18; // rsi
  __int64 v19; // rax
  int i; // ecx
  unsigned int v21; // esi
  unsigned int v22; // r14d
  unsigned int v23; // r14d
  __int64 v24; // rdx
  unsigned int v25; // r9d
  __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // r8
  unsigned __int64 *v29; // rsi
  unsigned __int64 v30; // r10
  int v31; // eax
  __int64 v32; // rdx
  int v33; // ecx
  __int64 v34; // rdi
  __int64 v35; // r8
  unsigned __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rdx
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // rdx
  unsigned int v41; // ecx
  unsigned __int64 v42; // r9
  unsigned int v43; // r10d
  int v44; // ecx
  unsigned int v45; // eax
  __int64 v46; // rdx
  unsigned int v47; // eax
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  unsigned int v50; // r8d
  int v51; // ecx
  unsigned int v52; // r10d
  int v53; // ecx
  __int64 v54; // rdx
  unsigned __int64 v55; // rdx
  __int64 v56; // [rsp+0h] [rbp-80h]
  unsigned int v57; // [rsp+8h] [rbp-78h]
  __int64 v58; // [rsp+8h] [rbp-78h]
  __int64 v59; // [rsp+8h] [rbp-78h]
  __int64 v60; // [rsp+10h] [rbp-70h]
  __int64 v61; // [rsp+10h] [rbp-70h]
  unsigned __int64 v62; // [rsp+10h] [rbp-70h]
  unsigned int v63; // [rsp+10h] [rbp-70h]
  unsigned __int64 v64; // [rsp+18h] [rbp-68h]
  unsigned __int64 v65; // [rsp+18h] [rbp-68h]
  unsigned int v66; // [rsp+18h] [rbp-68h]
  __int64 v67; // [rsp+18h] [rbp-68h]
  __int64 v68; // [rsp+20h] [rbp-60h]
  unsigned int v69; // [rsp+20h] [rbp-60h]
  __int64 v70; // [rsp+20h] [rbp-60h]
  unsigned int v71; // [rsp+20h] [rbp-60h]
  unsigned __int64 v72; // [rsp+20h] [rbp-60h]
  unsigned int v73; // [rsp+2Ch] [rbp-54h]
  __int64 v74; // [rsp+30h] [rbp-50h]
  __int64 v76; // [rsp+40h] [rbp-40h]
  __int64 *v77; // [rsp+48h] [rbp-38h]

  v3 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v3 )
  {
    v6 = *(__int64 **)a2;
    v74 = *(_QWORD *)a2 + 24 * v3;
    v8 = sub_1DBCA20(a1, a2);
    v9 = a2;
    if ( v8 )
    {
      v10 = (unsigned int *)(*(_QWORD *)(a1 + 592) + 8LL * *(unsigned int *)(v8 + 48));
      v11 = v10[1];
      v12 = 8LL * *v10;
      v77 = (__int64 *)(v12 + *(_QWORD *)(a1 + 432));
      v76 = *(_QWORD *)(a1 + 512) + v12;
    }
    else
    {
      v77 = *(__int64 **)(a1 + 432);
      v11 = *(unsigned int *)(a1 + 440);
      v76 = *(_QWORD *)(a1 + 512);
    }
    v13 = 8 * v11;
    v14 = v77;
    v15 = &v77[(unsigned __int64)v13 / 8];
    v16 = v13 >> 3;
    if ( v13 )
    {
      do
      {
        while ( 1 )
        {
          v17 = v16 >> 1;
          v18 = &v14[v16 >> 1];
          if ( (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) >= (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v6 >> 1) & 3) )
            break;
          v14 = v18 + 1;
          v16 = v16 - v17 - 1;
          if ( v16 <= 0 )
            goto LABEL_11;
        }
        v16 >>= 1;
      }
      while ( v17 > 0 );
    }
    else
    {
      v14 = v77;
    }
LABEL_11:
    if ( v15 != v14 )
    {
      v19 = *v14;
      v4 = 0;
      for ( i = *(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 24); ; i = *(_DWORD *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
      {
        while ( 1 )
        {
          v21 = i | (v19 >> 1) & 3;
          if ( v21 < (*(_DWORD *)((v6[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6[1] >> 1) & 3) )
            break;
          v37 = 24LL * *(unsigned int *)(v9 + 8);
          if ( v21 < (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v9 + v37 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                    | (unsigned int)(*(__int64 *)(*(_QWORD *)v9 + v37 - 16) >> 1) & 3) )
          {
            do
            {
              v38 = v6[4];
              v6 += 3;
            }
            while ( v21 >= (*(_DWORD *)((v38 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v38 >> 1) & 3) );
          }
          else
          {
            v6 = (__int64 *)(*(_QWORD *)v9 + v37);
          }
          if ( (__int64 *)v74 == v6 )
            return v4;
          while ( (i | (unsigned int)(v19 >> 1) & 3) < (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                      | (unsigned int)(*v6 >> 1) & 3) )
          {
            if ( ++v14 == v15 )
              return v4;
            v19 = *v14;
            i = *(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 24);
          }
        }
        if ( (_BYTE)v4 )
          goto LABEL_15;
        v39 = *(_QWORD *)(a3 + 8);
        *(_DWORD *)(a3 + 16) = 0;
        v40 = *(unsigned int *)(*(_QWORD *)(a1 + 248) + 16LL);
        v22 = *(_DWORD *)(*(_QWORD *)(a1 + 248) + 16LL);
        if ( v40 > v39 << 6 )
        {
          v45 = v40 + 63;
          v46 = 2 * v39;
          v56 = v9;
          v47 = v45 >> 6;
          v57 = v39;
          v73 = v47;
          v60 = v47;
          if ( 2 * v39 < v47 )
            v46 = v47;
          v64 = v46;
          v68 = 8 * v46;
          v48 = (__int64)realloc(*(_QWORD *)a3, 8 * v46, v46, 8 * (int)v46, v39, v47);
          v49 = v64;
          v42 = v60;
          v50 = v57;
          v9 = v56;
          if ( !v48 )
          {
            if ( v68 )
            {
              v59 = v60;
              v63 = v50;
              sub_16BD1C0("Allocation failed", 1u);
              v9 = v56;
              v42 = v59;
              v50 = v63;
              v48 = 0;
              v49 = v64;
            }
            else
            {
              v48 = malloc(1u);
              v49 = v64;
              v50 = v57;
              v42 = v60;
              v9 = v56;
              if ( !v48 )
              {
                sub_16BD1C0("Allocation failed", 1u);
                v49 = v64;
                v48 = 0;
                v50 = v57;
                v42 = v60;
                v9 = v56;
              }
            }
          }
          v51 = *(_DWORD *)(a3 + 16);
          *(_QWORD *)a3 = v48;
          *(_QWORD *)(a3 + 8) = v49;
          v52 = (unsigned int)(v51 + 63) >> 6;
          if ( v49 > v52 )
          {
            v55 = v49 - v52;
            if ( v55 )
            {
              v58 = v9;
              v62 = v42;
              v66 = v50;
              v71 = (unsigned int)(v51 + 63) >> 6;
              memset((void *)(v48 + 8LL * v52), 0, 8 * v55);
              v51 = *(_DWORD *)(a3 + 16);
              v48 = *(_QWORD *)a3;
              v9 = v58;
              v42 = v62;
              v50 = v66;
              v52 = v71;
            }
          }
          v53 = v51 & 0x3F;
          if ( v53 )
          {
            *(_QWORD *)(v48 + 8LL * (v52 - 1)) &= ~(-1LL << v53);
            v48 = *(_QWORD *)a3;
          }
          v54 = v50;
          v39 = *(_QWORD *)(a3 + 8);
          if ( v39 != v54 )
          {
            v67 = v9;
            v72 = v42;
            memset((void *)(v48 + 8 * v54), -1, 8 * (v39 - v54));
            v39 = *(_QWORD *)(a3 + 8);
            v9 = v67;
            v42 = v72;
          }
          v41 = *(_DWORD *)(a3 + 16);
        }
        else
        {
          v41 = 0;
          v73 = (unsigned int)(v40 + 63) >> 6;
          v42 = v73;
        }
        if ( v22 > v41 )
        {
          v43 = (v41 + 63) >> 6;
          if ( v43 < v39 )
          {
            v61 = v9;
            v65 = v42;
            v69 = (v41 + 63) >> 6;
            memset((void *)(*(_QWORD *)a3 + 8LL * v43), -1, 8 * (v39 - v43));
            v41 = *(_DWORD *)(a3 + 16);
            v9 = v61;
            v42 = v65;
            v43 = v69;
          }
          v44 = v41 & 0x3F;
          if ( v44 )
            *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v43 - 1)) |= -1LL << v44;
          v39 = *(_QWORD *)(a3 + 8);
        }
        *(_DWORD *)(a3 + 16) = v22;
        if ( v39 > v42 )
        {
          v70 = v9;
          memset((void *)(*(_QWORD *)a3 + 8 * v42), 0, 8 * (v39 - v42));
          v22 = *(_DWORD *)(a3 + 16);
          v9 = v70;
        }
        if ( (v22 & 0x3F) != 0 )
          break;
LABEL_16:
        v23 = v22 + 31;
        v24 = *(_QWORD *)(v76 + (char *)v14 - (char *)v77);
        v25 = v23 >> 5;
        if ( v23 <= 0x3F )
        {
          LODWORD(v27) = 0;
        }
        else
        {
          v26 = 0;
          v27 = ((v25 - 2) >> 1) + 1;
          v28 = 8 * v27;
          do
          {
            v29 = (unsigned __int64 *)(v26 + *(_QWORD *)a3);
            v30 = *v29 & ~(unsigned __int64)(unsigned int)~*(_DWORD *)(v24 + v26);
            v31 = *(_DWORD *)(v24 + v26 + 4);
            v26 += 8;
            *v29 = v30 & ~((unsigned __int64)(unsigned int)~v31 << 32);
          }
          while ( v28 != v26 );
          v24 += v28;
          v25 = (v23 & 0x20) != 0;
        }
        if ( v25 )
        {
          v32 = v24 + 4;
          v33 = 0;
          v34 = 8LL * (unsigned int)v27;
          v35 = v32;
          while ( 1 )
          {
            v36 = (unsigned __int64)(unsigned int)~*(_DWORD *)(v32 - 4) << v33;
            v33 += 32;
            *(_QWORD *)(v34 + *(_QWORD *)a3) &= ~v36;
            if ( v32 == v35 )
              break;
            v32 += 4;
          }
        }
        if ( v14 + 1 == v15 )
          return 1;
        v19 = v14[1];
        v4 = 1;
        ++v14;
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v73 - 1)) &= ~(-1LL << (v22 & 0x3F));
LABEL_15:
      v22 = *(_DWORD *)(a3 + 16);
      goto LABEL_16;
    }
  }
  return 0;
}
