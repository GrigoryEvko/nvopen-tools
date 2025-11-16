// Function: sub_20FD0B0
// Address: 0x20fd0b0
//
__int64 __fastcall sub_20FD0B0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 result; // rax
  int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 *v13; // r10
  __int64 v14; // r13
  __int64 v15; // r11
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rcx
  int v19; // r15d
  __int64 v20; // rcx
  unsigned int v21; // edx
  unsigned int v22; // eax
  __int64 v23; // rcx
  unsigned int v24; // r15d
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // r12
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // r12
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
  unsigned int v47; // ecx
  __int64 v48; // r10
  unsigned int v49; // edx
  __int64 v50; // r12
  int v51; // r8d
  __int64 i; // rax
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // [rsp-50h] [rbp-50h]
  unsigned int v57; // [rsp-44h] [rbp-44h]
  __int64 v58; // [rsp-40h] [rbp-40h]
  __int64 v59; // [rsp-40h] [rbp-40h]

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
        sub_20FCFA0(a1 + 24, v37, v33, v36, a5, a6);
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
          v59 = v42;
          sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 16, v42, a6);
          v42 = v59;
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
    v58 = 0;
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
      v16 = *(_QWORD *)v15;
      v17 = 16LL * *(unsigned int *)(v15 + 12);
      v18 = *(_QWORD *)(*(_QWORD *)v15 + v17);
      v19 = *(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v20 = v18 >> 1;
      v21 = *(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v13 >> 1) & 3;
      v22 = *(_DWORD *)((*(_QWORD *)(*(_QWORD *)v15 + v17 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (*(__int64 *)(*(_QWORD *)v15 + v17 + 8) >> 1) & 3;
      if ( *(_DWORD *)(v14 + 192) )
      {
        if ( v21 >= v22 )
          goto LABEL_56;
        v23 = v20 & 3;
        v24 = v23 | v19;
        v25 = v13[1];
        v26 = v25 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_DWORD *)((v25 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v25 >> 1) & 3) <= v24 )
          goto LABEL_34;
      }
      else
      {
        if ( v21 >= v22 )
        {
LABEL_56:
          v24 = v20 & 3 | v19;
LABEL_34:
          v45 = *(_QWORD *)(a1 + 8);
          v46 = 24LL * *(unsigned int *)(v45 + 8);
          if ( (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v45 + v46 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
              | (unsigned int)(*(__int64 *)(*(_QWORD *)v45 + v46 - 16) >> 1) & 3) > v24 )
          {
            if ( (*(_DWORD *)((v13[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13[1] >> 1) & 3) <= v24 )
            {
              do
              {
                v55 = v13[4];
                v13 += 3;
              }
              while ( (*(_DWORD *)((v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v55 >> 1) & 3) <= v24 );
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
              sub_20F82D0(a1 + 24, v48);
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
        v23 = v20 & 3;
        v24 = v23 | v19;
        v44 = v13[1];
        v26 = v44 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v24 >= (*(_DWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v44 >> 1) & 3) )
          goto LABEL_34;
      }
      v27 = *(_QWORD *)(v16 + 8LL * *(unsigned int *)(v15 + 12) + 128);
      if ( v27 != v58 )
      {
        if ( (unsigned __int8)sub_20FC340(a1, *(_QWORD *)(v16 + 8LL * *(unsigned int *)(v15 + 12) + 128), v26, v23, v11) )
        {
          v10 = *(_QWORD *)(a1 + 32);
          v12 = *(unsigned int *)(a1 + 40);
        }
        else
        {
          v54 = *(unsigned int *)(a1 + 120);
          if ( (unsigned int)v54 >= *(_DWORD *)(a1 + 124) )
          {
            sub_16CD150(a1 + 112, (const void *)(a1 + 128), 0, 8, v28, v29);
            v54 = *(unsigned int *)(a1 + 120);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 112) + 8 * v54) = v27;
          result = (unsigned int)(*(_DWORD *)(a1 + 120) + 1);
          *(_DWORD *)(a1 + 120) = result;
          if ( v57 <= (unsigned int)result )
            return result;
          v10 = *(_QWORD *)(a1 + 32);
          v12 = *(unsigned int *)(a1 + 40);
          v58 = v27;
        }
      }
      v30 = v10 + 16 * v12 - 16;
      v31 = *(_DWORD *)(v30 + 12) + 1;
      *(_DWORD *)(v30 + 12) = v31;
      v11 = *(unsigned int *)(a1 + 40);
      if ( v31 == *(_DWORD *)(*(_QWORD *)(a1 + 32) + 16 * v11 - 8) )
      {
        v53 = *(unsigned int *)(*(_QWORD *)(a1 + 24) + 192LL);
        if ( (_DWORD)v53 )
        {
          sub_39460A0(a1 + 32, v53);
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
