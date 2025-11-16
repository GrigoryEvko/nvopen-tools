// Function: sub_2EF9210
// Address: 0x2ef9210
//
__int64 __fastcall sub_2EF9210(__int64 a1, __int64 a2, __int64 a3)
{
  int *v3; // r13
  unsigned int v4; // r11d
  __int64 v5; // r9
  int *v6; // r14
  __int64 result; // rax
  __int64 v10; // r8
  unsigned int v11; // r10d
  int v12; // r15d
  unsigned int v13; // eax
  unsigned __int64 v14; // rdx
  int v15; // ecx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r11
  int v18; // r10d
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // r14
  int v25; // ecx
  unsigned int v26; // esi
  __int64 v27; // r9
  int v28; // r15d
  unsigned int v29; // edx
  _DWORD *v30; // rdi
  int v31; // r8d
  int v32; // r11d
  _DWORD *v33; // rax
  int v34; // edi
  int v35; // edx
  int v36; // eax
  __int64 v37; // rcx
  unsigned int v38; // edx
  int v39; // esi
  int v40; // edi
  __int64 v41; // r13
  int v42; // r9d
  int v43; // r9d
  __int64 v44; // r10
  _DWORD *v45; // rsi
  __int64 v46; // r15
  int v47; // edi
  int v48; // r8d
  int v49; // r9d
  int v50; // r9d
  __int64 v51; // r10
  __int64 v52; // rsi
  int v53; // r15d
  int v54; // r8d
  _DWORD *v55; // rdi
  unsigned int v56; // [rsp+8h] [rbp-48h]
  unsigned int v57; // [rsp+Ch] [rbp-44h]
  const void *v58; // [rsp+10h] [rbp-40h]
  __int64 v59; // [rsp+10h] [rbp-40h]
  int v60; // [rsp+10h] [rbp-40h]
  int v61; // [rsp+10h] [rbp-40h]
  int v62; // [rsp+18h] [rbp-38h]
  unsigned int v63; // [rsp+18h] [rbp-38h]
  int v64; // [rsp+18h] [rbp-38h]
  int v65; // [rsp+18h] [rbp-38h]
  unsigned int v66; // [rsp+1Ch] [rbp-34h]
  int v67; // [rsp+1Ch] [rbp-34h]
  int v68; // [rsp+1Ch] [rbp-34h]
  int v69; // [rsp+1Ch] [rbp-34h]

  v3 = *(int **)(a2 + 8);
  v4 = *(_DWORD *)(a1 + 64);
  v5 = *(unsigned int *)(a3 + 8);
  v66 = *(_DWORD *)(a1 + 88);
  v6 = &v3[*(unsigned int *)(a2 + 24)];
  result = *(unsigned int *)(a2 + 16);
  if ( (_DWORD)result && v3 != v6 )
  {
    while ( (unsigned int)*v3 > 0xFFFFFFFD )
    {
      if ( ++v3 == v6 )
        return result;
    }
    if ( v3 != v6 )
    {
      result = a3 + 16;
      v10 = (unsigned int)v5;
      v11 = *(_DWORD *)(a1 + 64);
      v58 = (const void *)(a3 + 16);
      do
      {
        v12 = *v3;
        if ( *v3 >= 0 )
          goto LABEL_18;
        result = v12 & 0x7FFFFFFF;
        if ( (unsigned int)result > 0x13FFF )
        {
          v36 = *(_DWORD *)(a1 + 96);
          v37 = *(_QWORD *)(a1 + 80);
          if ( v36 )
          {
            result = (unsigned int)(v36 - 1);
            v38 = result & (37 * v12);
            v39 = *(_DWORD *)(v37 + 4LL * v38);
            if ( v12 == v39 )
              goto LABEL_18;
            v40 = 1;
            while ( v39 != -1 )
            {
              v38 = result & (v40 + v38);
              v39 = *(_DWORD *)(v37 + 4LL * v38);
              if ( v39 == v12 )
                goto LABEL_18;
              ++v40;
            }
          }
          v14 = v10 + 1;
          ++v66;
          if ( v10 + 1 <= (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
LABEL_16:
            *(_DWORD *)(*(_QWORD *)a3 + 4 * v10) = v12;
            result = *(unsigned int *)(a3 + 8);
            v10 = (unsigned int)(result + 1);
            *(_DWORD *)(a3 + 8) = v10;
            goto LABEL_18;
          }
LABEL_54:
          v56 = v11;
          v57 = v5;
          v63 = v4;
          sub_C8D5F0(a3, v58, v14, 4u, v10, v5);
          v10 = *(unsigned int *)(a3 + 8);
          v11 = v56;
          v5 = v57;
          v4 = v63;
          goto LABEL_16;
        }
        if ( v4 <= (unsigned int)result
          || (*(_QWORD *)(*(_QWORD *)a1 + 8LL * ((unsigned int)result >> 6)) & (1LL << v12)) == 0 )
        {
          v13 = result + 1;
          v14 = v10 + 1;
          if ( v11 < v13 )
            v11 = v13;
          if ( v14 <= *(unsigned int *)(a3 + 12) )
            goto LABEL_16;
          goto LABEL_54;
        }
        do
        {
LABEL_18:
          if ( ++v3 == v6 )
            goto LABEL_19;
        }
        while ( (unsigned int)*v3 > 0xFFFFFFFD );
      }
      while ( v3 != v6 );
LABEL_19:
      if ( (_DWORD)v5 != (_DWORD)v10 )
      {
        v15 = *(_DWORD *)(a1 + 64) & 0x3F;
        if ( v15 )
          *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v15);
        v16 = *(unsigned int *)(a1 + 8);
        *(_DWORD *)(a1 + 64) = v11;
        v17 = (v11 + 63) >> 6;
        if ( v17 != v16 )
        {
          if ( v17 >= v16 )
          {
            v41 = v17 - v16;
            if ( v17 > *(unsigned int *)(a1 + 12) )
            {
              v65 = v10;
              v61 = v5;
              sub_C8D5F0(a1, (const void *)(a1 + 16), (v11 + 63) >> 6, 8u, v10, v5);
              v16 = *(unsigned int *)(a1 + 8);
              LODWORD(v10) = v65;
              LODWORD(v5) = v61;
            }
            if ( 8 * v41 )
            {
              v64 = v10;
              v60 = v5;
              memset((void *)(*(_QWORD *)a1 + 8 * v16), 0, 8 * v41);
              LODWORD(v16) = *(_DWORD *)(a1 + 8);
              LODWORD(v10) = v64;
              LODWORD(v5) = v60;
            }
            v11 = *(_DWORD *)(a1 + 64);
            *(_DWORD *)(a1 + 8) = v41 + v16;
          }
          else
          {
            *(_DWORD *)(a1 + 8) = (v11 + 63) >> 6;
          }
        }
        v18 = v11 & 0x3F;
        if ( v18 )
          *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v18);
        v59 = a1 + 72;
        v19 = *(_QWORD *)(a1 + 72) + 1LL;
        result = v66;
        if ( v66 )
        {
          *(_QWORD *)(a1 + 72) = v19;
          v20 = (4 * v66 / 3 + 1) | ((unsigned __int64)(4 * v66 / 3 + 1) >> 1);
          v21 = (((v20 >> 2) | v20) >> 4) | (v20 >> 2) | v20;
          result = (((v21 >> 8) | v21) >> 16) | (v21 >> 8) | v21;
          if ( *(_DWORD *)(a1 + 96) < (unsigned int)(result + 1) )
          {
            v62 = v10;
            v67 = v5;
            result = (__int64)sub_2E29BA0(a1 + 72, (int)result + 1);
            LODWORD(v10) = v62;
            LODWORD(v5) = v67;
          }
        }
        else
        {
          *(_QWORD *)(a1 + 72) = v19;
        }
        v22 = (unsigned int)v5;
        if ( (unsigned int)v5 < (unsigned int)v10 )
        {
          v23 = (unsigned int)(v5 + 1);
          v24 = v23 + (unsigned int)(~(_DWORD)v5 + v10) + 1;
          while ( 1 )
          {
            v25 = *(_DWORD *)(*(_QWORD *)a3 + 4 * v22);
            if ( (v25 & 0x7FFFFFFFu) <= 0x13FFF )
            {
              *(_QWORD *)(*(_QWORD *)a1 + 8LL * ((v25 & 0x7FFFFFFFu) >> 6)) |= 1LL << v25;
              goto LABEL_33;
            }
            v26 = *(_DWORD *)(a1 + 96);
            if ( !v26 )
              break;
            v27 = *(_QWORD *)(a1 + 80);
            v28 = 37 * v25;
            v29 = (v26 - 1) & (37 * v25);
            v30 = (_DWORD *)(v27 + 4LL * v29);
            v31 = *v30;
            if ( *v30 == v25 )
              goto LABEL_33;
            v32 = 1;
            v33 = 0;
            while ( v31 != -1 )
            {
              if ( !v33 && v31 == -2 )
                v33 = v30;
              v29 = (v26 - 1) & (v32 + v29);
              v30 = (_DWORD *)(v27 + 4LL * v29);
              v31 = *v30;
              if ( *v30 == v25 )
                goto LABEL_33;
              ++v32;
            }
            if ( !v33 )
              v33 = v30;
            v34 = *(_DWORD *)(a1 + 88);
            ++*(_QWORD *)(a1 + 72);
            v35 = v34 + 1;
            if ( 4 * (v34 + 1) >= 3 * v26 )
              goto LABEL_68;
            if ( v26 - *(_DWORD *)(a1 + 92) - v35 <= v26 >> 3 )
            {
              v68 = v25;
              sub_2E29BA0(v59, v26);
              v42 = *(_DWORD *)(a1 + 96);
              if ( !v42 )
                goto LABEL_92;
              v43 = v42 - 1;
              v44 = *(_QWORD *)(a1 + 80);
              v45 = 0;
              LODWORD(v46) = v43 & v28;
              v25 = v68;
              v35 = *(_DWORD *)(a1 + 88) + 1;
              v47 = 1;
              v33 = (_DWORD *)(v44 + 4LL * (unsigned int)v46);
              v48 = *v33;
              if ( *v33 != v68 )
              {
                while ( v48 != -1 )
                {
                  if ( v48 == -2 && !v45 )
                    v45 = v33;
                  v46 = v43 & (unsigned int)(v46 + v47);
                  v33 = (_DWORD *)(v44 + 4 * v46);
                  v48 = *v33;
                  if ( *v33 == v68 )
                    goto LABEL_44;
                  ++v47;
                }
                if ( v45 )
                  v33 = v45;
              }
            }
LABEL_44:
            *(_DWORD *)(a1 + 88) = v35;
            if ( *v33 != -1 )
              --*(_DWORD *)(a1 + 92);
            *v33 = v25;
LABEL_33:
            result = v23 + 1;
            v22 = v23;
            if ( v24 == v23 + 1 )
              return result;
            ++v23;
          }
          ++*(_QWORD *)(a1 + 72);
LABEL_68:
          v69 = v25;
          sub_2E29BA0(v59, 2 * v26);
          v49 = *(_DWORD *)(a1 + 96);
          if ( !v49 )
          {
LABEL_92:
            ++*(_DWORD *)(a1 + 88);
            BUG();
          }
          v25 = v69;
          v50 = v49 - 1;
          v51 = *(_QWORD *)(a1 + 80);
          v35 = *(_DWORD *)(a1 + 88) + 1;
          LODWORD(v52) = v50 & (37 * v69);
          v33 = (_DWORD *)(v51 + 4LL * (unsigned int)v52);
          v53 = *v33;
          if ( v69 != *v33 )
          {
            v54 = 1;
            v55 = 0;
            while ( v53 != -1 )
            {
              if ( !v55 && v53 == -2 )
                v55 = v33;
              v52 = v50 & (unsigned int)(v52 + v54);
              v33 = (_DWORD *)(v51 + 4 * v52);
              v53 = *v33;
              if ( *v33 == v69 )
                goto LABEL_44;
              ++v54;
            }
            if ( v55 )
              v33 = v55;
          }
          goto LABEL_44;
        }
      }
    }
  }
  return result;
}
