// Function: sub_2E1AC90
// Address: 0x2e1ac90
//
__int64 __fastcall sub_2E1AC90(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // r10
  __int64 *v13; // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r11
  __int64 v18; // rdx
  __int64 v19; // rdi
  int v20; // r15d
  unsigned int v21; // edx
  unsigned int v22; // eax
  unsigned int v23; // r15d
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // r8
  __int64 *v39; // rdx
  int v40; // r15d
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned int v47; // edi
  __int64 v48; // r8
  unsigned int v49; // edx
  __int64 v50; // r11
  int v51; // esi
  __int64 i; // rax
  unsigned int v53; // r8d
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // [rsp-50h] [rbp-50h]
  unsigned int v57; // [rsp-48h] [rbp-48h]
  int v58; // [rsp-44h] [rbp-44h]
  __int64 v59; // [rsp-40h] [rbp-40h]
  __int64 v60; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 120);
  if ( !*(_BYTE *)(a1 + 161) && a2 > (unsigned int)result )
  {
    if ( *(_BYTE *)(a1 + 160) )
    {
      v9 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      v32 = *(_QWORD *)(a1 + 8);
      *(_BYTE *)(a1 + 160) = 1;
      if ( !*(_DWORD *)(v32 + 8) || (v33 = *(_QWORD *)a1, !*(_DWORD *)(*(_QWORD *)a1 + 204LL)) )
      {
        *(_BYTE *)(a1 + 161) = 1;
        return 0;
      }
      v34 = *(__int64 **)v32;
      v35 = v33 + 8;
      *(_QWORD *)(a1 + 24) = v33 + 8;
      *(_QWORD *)(a1 + 16) = v34;
      v36 = *(unsigned int *)(v33 + 200);
      v37 = *v34;
      if ( (_DWORD)v36 )
      {
        sub_2E1A860(a1 + 24, v37, v33, v36, a5, a6);
        v9 = *(_DWORD *)(a1 + 40);
      }
      else
      {
        v38 = *(unsigned int *)(v33 + 204);
        if ( (_DWORD)v38 )
        {
          v39 = (__int64 *)(v33 + 16);
          do
          {
            if ( (*(_DWORD *)((*v39 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v39 >> 1) & 3) > (*(_DWORD *)((v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v37 >> 1) & 3) )
              break;
            v36 = (unsigned int)(v36 + 1);
            v39 += 2;
          }
          while ( (_DWORD)v38 != (_DWORD)v36 );
        }
        v40 = *(_DWORD *)(a1 + 44);
        *(_DWORD *)(a1 + 40) = 0;
        v41 = 0;
        v42 = (v36 << 32) | v38;
        if ( !v40 )
        {
          v60 = v42;
          sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), 1u, 0x10u, v42, a6);
          v42 = v60;
          v41 = 16LL * *(unsigned int *)(a1 + 40);
        }
        v43 = *(_QWORD *)(a1 + 32);
        *(_QWORD *)(v43 + v41) = v35;
        *(_QWORD *)(v43 + v41 + 8) = v42;
        v9 = *(_DWORD *)(a1 + 40) + 1;
        *(_DWORD *)(a1 + 40) = v9;
      }
    }
    if ( !v9 )
      goto LABEL_19;
    v59 = 0;
    v57 = a2;
    v56 = **(_QWORD **)(a1 + 8) + 24LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 8LL);
LABEL_8:
    v10 = *(_QWORD *)(a1 + 32);
    if ( *(_DWORD *)(v10 + 12) >= *(_DWORD *)(v10 + 8) )
      goto LABEL_19;
    LODWORD(v11) = *(_DWORD *)(a1 + 40);
    while ( 1 )
    {
      v12 = (unsigned int)v11;
      v13 = *(__int64 **)(a1 + 16);
      v14 = *(_QWORD *)(a1 + 24);
      v15 = v10 + 16LL * (unsigned int)v11 - 16;
      v16 = *(unsigned int *)(v15 + 12);
      v17 = *(_QWORD *)v15;
      v18 = *(_QWORD *)(*(_QWORD *)v15 + 16 * v16);
      v19 = (v18 >> 1) & 3;
      v20 = *(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v58 = (v18 >> 1) & 3;
      v21 = *(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v13 >> 1) & 3;
      v22 = *(_DWORD *)((*(_QWORD *)(*(_QWORD *)v15 + 16 * v16 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (*(__int64 *)(*(_QWORD *)v15 + 16 * v16 + 8) >> 1) & 3;
      if ( *(_DWORD *)(v14 + 192) )
      {
        if ( v21 >= v22 )
          goto LABEL_56;
        v23 = v19 | v20;
        v24 = v13[1];
        v25 = v24 >> 1;
        v26 = v24 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v23 >= (*(_DWORD *)(v26 + 24) | (unsigned int)(v25 & 3)) )
          goto LABEL_34;
      }
      else
      {
        if ( v21 >= v22 )
        {
LABEL_56:
          v23 = v58 | v20;
LABEL_34:
          v45 = *(_QWORD *)(a1 + 8);
          v46 = 24LL * *(unsigned int *)(v45 + 8);
          if ( (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v45 + v46 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
              | (unsigned int)(*(__int64 *)(*(_QWORD *)v45 + v46 - 16) >> 1) & 3) > v23 )
          {
            if ( (*(_DWORD *)((v13[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13[1] >> 1) & 3) <= v23 )
            {
              do
              {
                v55 = v13[4];
                v13 += 3;
              }
              while ( (*(_DWORD *)((v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v55 >> 1) & 3) <= v23 );
            }
          }
          else
          {
            v13 = (__int64 *)(*(_QWORD *)v45 + v46);
          }
          *(_QWORD *)(a1 + 16) = v13;
          if ( (__int64 *)v56 == v13 )
            goto LABEL_19;
          v47 = *(_DWORD *)(v15 + 12);
          v48 = *v13;
          v49 = *(_DWORD *)((v48 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v48 >> 1) & 3;
          v50 = *(_QWORD *)(*(_QWORD *)v15 + 16LL * v47 + 8);
          if ( v49 < (*(_DWORD *)((v50 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v50 >> 1) & 3) )
            goto LABEL_47;
          if ( !(_DWORD)v11 )
            goto LABEL_19;
          if ( *(_DWORD *)(v10 + 12) < *(_DWORD *)(v10 + 8) )
          {
            if ( *(_DWORD *)(v14 + 192) )
            {
              sub_2E1A970(a1 + 24, v48);
              LODWORD(v11) = *(_DWORD *)(a1 + 40);
            }
            else
            {
              v51 = *(_DWORD *)(v14 + 196);
              if ( v51 != v47 )
              {
                for ( i = *(unsigned int *)(v15 + 12);
                      v49 >= (*(_DWORD *)((*(_QWORD *)(v14 + 16 * i + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                            | (unsigned int)(*(__int64 *)(v14 + 16 * i + 8) >> 1) & 3);
                      i = v47 )
                {
                  if ( v51 == ++v47 )
                    break;
                }
              }
              *(_DWORD *)(v15 + 12) = v47;
              LODWORD(v11) = *(_DWORD *)(a1 + 40);
            }
          }
LABEL_47:
          if ( !(_DWORD)v11 )
          {
LABEL_19:
            *(_BYTE *)(a1 + 161) = 1;
            return *(unsigned int *)(a1 + 120);
          }
          goto LABEL_8;
        }
        v23 = v19 | v20;
        v44 = v13[1];
        v26 = v44 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v23 >= (*(_DWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v44 >> 1) & 3) )
          goto LABEL_34;
      }
      v27 = *(_QWORD *)(v17 + 8 * v16 + 128);
      if ( v59 != v27 )
      {
        if ( (unsigned __int8)sub_2E19A00(a1, *(_QWORD *)(v17 + 8 * v16 + 128), v26, v10, (unsigned int)v13) )
        {
          v10 = *(_QWORD *)(a1 + 32);
          v12 = *(unsigned int *)(a1 + 40);
        }
        else
        {
          v54 = *(unsigned int *)(a1 + 120);
          if ( v54 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 124) )
          {
            sub_C8D5F0(a1 + 112, (const void *)(a1 + 128), v54 + 1, 8u, v28, v29);
            v54 = *(unsigned int *)(a1 + 120);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 112) + 8 * v54) = v27;
          result = (unsigned int)(*(_DWORD *)(a1 + 120) + 1);
          *(_DWORD *)(a1 + 120) = result;
          if ( v57 <= (unsigned int)result )
            return result;
          v10 = *(_QWORD *)(a1 + 32);
          v12 = *(unsigned int *)(a1 + 40);
          v59 = v27;
        }
      }
      v30 = v10 + 16 * v12 - 16;
      v31 = *(_DWORD *)(v30 + 12) + 1;
      *(_DWORD *)(v30 + 12) = v31;
      v11 = *(unsigned int *)(a1 + 40);
      if ( v31 == *(_DWORD *)(*(_QWORD *)(a1 + 32) + 16 * v11 - 8) )
      {
        v53 = *(_DWORD *)(*(_QWORD *)(a1 + 24) + 192LL);
        if ( v53 )
        {
          sub_F03D40((__int64 *)(a1 + 32), v53);
          LODWORD(v11) = *(_DWORD *)(a1 + 40);
        }
      }
      if ( (_DWORD)v11 )
      {
        v10 = *(_QWORD *)(a1 + 32);
        if ( *(_DWORD *)(v10 + 12) < *(_DWORD *)(v10 + 8) )
          continue;
      }
      goto LABEL_19;
    }
  }
  return result;
}
