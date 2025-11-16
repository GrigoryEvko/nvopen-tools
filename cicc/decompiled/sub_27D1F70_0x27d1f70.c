// Function: sub_27D1F70
// Address: 0x27d1f70
//
char __fastcall sub_27D1F70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  int v9; // eax
  int v10; // edx
  __int64 v11; // r8
  _QWORD *v12; // r9
  int v13; // r10d
  _QWORD *v14; // rsi
  __int64 v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  int v22; // eax
  unsigned __int64 *v23; // rcx
  __int64 v24; // r15
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 v27; // r12
  __int64 v28; // rdx
  __int64 v29; // rbx
  int v30; // eax
  int v31; // edx
  __int64 v32; // rdi
  int v33; // r9d
  _QWORD *v34; // r8
  _QWORD *v35; // rsi
  __int64 v36; // rcx
  _QWORD *v37; // rax
  __int64 v38; // r9
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rcx
  int v41; // eax
  unsigned __int64 *v42; // rdx
  int v43; // edx
  __int64 v44; // rsi
  int v45; // edx
  int v46; // r8d
  _QWORD *v47; // rdi
  _QWORD *v48; // r10
  __int64 v49; // rcx
  _QWORD *v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rcx
  int v55; // eax
  unsigned __int64 *v56; // rdx
  unsigned __int64 v57; // r14
  unsigned __int64 v58; // r14
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // rbx
  __int64 v62; // [rsp+8h] [rbp-58h]
  __int64 v63; // [rsp+10h] [rbp-50h]
  __int64 v64; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v65[7]; // [rsp+28h] [rbp-38h] BYREF

  v64 = a2;
  if ( *(_BYTE *)a2 == 5 )
  {
    LOBYTE(v8) = sub_27CE4B0(a2, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24));
    if ( (_BYTE)v8 )
    {
      v30 = *(_DWORD *)(a4 + 24);
      v65[0] = a2;
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a4 + 8);
        v33 = 1;
        v34 = 0;
        LODWORD(v8) = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v35 = (_QWORD *)(v32 + 8LL * (unsigned int)v8);
        v36 = *v35;
        if ( a2 == *v35 )
          return v8;
        while ( v36 != -4096 )
        {
          if ( !v34 && v36 == -8192 )
            v34 = v35;
          LODWORD(v8) = v31 & (v33 + v8);
          v35 = (_QWORD *)(v32 + 8LL * (unsigned int)v8);
          v36 = *v35;
          if ( a2 == *v35 )
            return v8;
          ++v33;
        }
        if ( !v34 )
          v34 = v35;
      }
      else
      {
        v34 = 0;
      }
      v37 = sub_27D1DE0(a4, v65, v34);
      *v37 = v65[0];
      v39 = *(unsigned int *)(a3 + 8);
      v40 = *(unsigned int *)(a3 + 12);
      v41 = *(_DWORD *)(a3 + 8);
      if ( v39 >= v40 )
      {
        v57 = a2 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v40 < v39 + 1 )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v39 + 1, 8u, v39 + 1, v38);
          v39 = *(unsigned int *)(a3 + 8);
        }
        v8 = *(_QWORD *)a3;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v39) = v57;
        ++*(_DWORD *)(a3 + 8);
      }
      else
      {
        v42 = (unsigned __int64 *)(*(_QWORD *)a3 + 8 * v39);
        if ( v42 )
        {
          *v42 = a2 & 0xFFFFFFFFFFFFFFFBLL;
          v41 = *(_DWORD *)(a3 + 8);
        }
        LODWORD(v8) = v41 + 1;
        *(_DWORD *)(a3 + 8) = v8;
      }
    }
    return v8;
  }
  v8 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  LODWORD(v8) = *(_DWORD *)(v8 + 8) >> 8;
  if ( *(_DWORD *)(a1 + 40) == (_DWORD)v8 )
  {
    LOBYTE(v8) = sub_27CE4B0(a2, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24));
    if ( (_BYTE)v8 )
    {
      v9 = *(_DWORD *)(a4 + 24);
      if ( v9 )
      {
        v10 = v9 - 1;
        v11 = *(_QWORD *)(a4 + 8);
        v12 = 0;
        v13 = 1;
        LODWORD(v8) = (v9 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
        v14 = (_QWORD *)(v11 + 8LL * (unsigned int)v8);
        v15 = *v14;
        if ( v64 == *v14 )
          return v8;
        while ( v15 != -4096 )
        {
          if ( v15 == -8192 && !v12 )
            v12 = v14;
          LODWORD(v8) = v10 & (v13 + v8);
          v14 = (_QWORD *)(v11 + 8LL * (unsigned int)v8);
          v15 = *v14;
          if ( v64 == *v14 )
            return v8;
          ++v13;
        }
        if ( !v12 )
          v12 = v14;
      }
      else
      {
        v12 = 0;
      }
      v16 = sub_27D1DE0(a4, &v64, v12);
      v19 = v64;
      *v16 = v64;
      v20 = *(unsigned int *)(a3 + 8);
      v21 = *(unsigned int *)(a3 + 12);
      v22 = *(_DWORD *)(a3 + 8);
      if ( v20 >= v21 )
      {
        v58 = v19 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v21 < v20 + 1 )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v20 + 1, 8u, v17, v18);
          v20 = *(unsigned int *)(a3 + 8);
        }
        v8 = *(_QWORD *)a3;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v20) = v58;
        ++*(_DWORD *)(a3 + 8);
      }
      else
      {
        v23 = (unsigned __int64 *)(*(_QWORD *)a3 + 8 * v20);
        if ( v23 )
        {
          *v23 = v19 & 0xFFFFFFFFFFFFFFFBLL;
          v22 = *(_DWORD *)(a3 + 8);
        }
        LODWORD(v8) = v22 + 1;
        *(_DWORD *)(a3 + 8) = v8;
      }
      v24 = v64;
      if ( *(_BYTE *)v64 != 22 )
      {
        LODWORD(v8) = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
        if ( (_DWORD)v8 )
        {
          v62 = a3;
          v25 = 0;
          v8 = 32LL * (unsigned int)v8;
          v63 = a4;
          v26 = a1;
          v27 = v8;
          do
          {
            if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
            {
              v28 = *(_QWORD *)(v24 - 8);
            }
            else
            {
              v8 = 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF);
              v28 = v24 - v8;
            }
            v29 = *(_QWORD *)(v28 + v25);
            if ( *(_BYTE *)v29 == 5 )
            {
              LOBYTE(v8) = sub_27CE4B0(v29, *(_QWORD *)(v26 + 32), *(_QWORD *)(v26 + 24));
              if ( (_BYTE)v8 )
              {
                v65[0] = v29;
                v43 = *(_DWORD *)(v63 + 24);
                if ( !v43 )
                {
                  v48 = 0;
LABEL_40:
                  v50 = sub_27D1DE0(v63, v65, v48);
                  *v50 = v65[0];
                  v53 = *(unsigned int *)(v62 + 8);
                  v54 = *(unsigned int *)(v62 + 12);
                  v55 = *(_DWORD *)(v62 + 8);
                  if ( v53 >= v54 )
                  {
                    v59 = v53 + 1;
                    v60 = v29 & 0xFFFFFFFFFFFFFFFBLL;
                    if ( v54 < v59 )
                      sub_C8D5F0(v62, (const void *)(v62 + 16), v59, 8u, v51, v52);
                    v8 = *(_QWORD *)v62;
                    *(_QWORD *)(*(_QWORD *)v62 + 8LL * (unsigned int)(*(_DWORD *)(v62 + 8))++) = v60;
                  }
                  else
                  {
                    v56 = (unsigned __int64 *)(*(_QWORD *)v62 + 8 * v53);
                    if ( v56 )
                    {
                      *v56 = v29 & 0xFFFFFFFFFFFFFFFBLL;
                      v55 = *(_DWORD *)(v62 + 8);
                    }
                    LODWORD(v8) = v55 + 1;
                    *(_DWORD *)(v62 + 8) = v8;
                  }
                  goto LABEL_22;
                }
                v44 = *(_QWORD *)(v63 + 8);
                v45 = v43 - 1;
                v46 = 1;
                v47 = 0;
                LODWORD(v8) = v45 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                v48 = (_QWORD *)(v44 + 8LL * (unsigned int)v8);
                v49 = *v48;
                if ( *v48 != v29 )
                {
                  while ( v49 != -4096 )
                  {
                    if ( v49 == -8192 && !v47 )
                      v47 = v48;
                    LODWORD(v8) = v45 & (v46 + v8);
                    v48 = (_QWORD *)(v44 + 8LL * (unsigned int)v8);
                    v49 = *v48;
                    if ( v29 == *v48 )
                      goto LABEL_22;
                    ++v46;
                  }
                  if ( v47 )
                    v48 = v47;
                  goto LABEL_40;
                }
              }
            }
LABEL_22:
            v25 += 32;
          }
          while ( v27 != v25 );
        }
      }
    }
  }
  return v8;
}
