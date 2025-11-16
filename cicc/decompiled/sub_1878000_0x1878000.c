// Function: sub_1878000
// Address: 0x1878000
//
int *__fastcall sub_1878000(__int64 a1, int *a2, int *a3)
{
  int *result; // rax
  __int64 v4; // r12
  int *v5; // r15
  __int64 v6; // r14
  __int64 v7; // rcx
  int v8; // esi
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rcx
  bool v12; // zf
  __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rcx
  int v17; // esi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 v23; // rcx
  int v24; // esi
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 v29; // rcx
  __int64 v30; // rbx
  __int64 v31; // rdi
  __int64 v32; // rcx
  int v33; // esi
  __int64 v34; // rax
  int *v35; // r13
  int *v36; // rbx
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // rdi
  __int64 v44; // rax
  int v45; // edx
  int *v46; // rax
  __int64 v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+10h] [rbp-A0h]
  signed __int64 v49; // [rsp+18h] [rbp-98h]
  signed __int64 v50; // [rsp+20h] [rbp-90h]
  __int64 v51; // [rsp+28h] [rbp-88h]
  int v52; // [rsp+38h] [rbp-78h] BYREF
  __int64 v53; // [rsp+40h] [rbp-70h]
  int *v54; // [rsp+48h] [rbp-68h]
  int *v55; // [rsp+50h] [rbp-60h]
  __int64 i; // [rsp+58h] [rbp-58h]
  __int64 v57; // [rsp+60h] [rbp-50h]
  __int64 v58; // [rsp+68h] [rbp-48h]
  __int64 v59; // [rsp+70h] [rbp-40h]
  __int64 v60; // [rsp+78h] [rbp-38h]

  result = a3;
  v48 = a1;
  if ( (int *)a1 != a2 )
  {
    result = (int *)a1;
    if ( a2 != a3 )
    {
      v47 = a1 + (char *)a3 - (char *)a2;
      v49 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)a3 - a1) >> 4);
      v51 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)a2 - a1) >> 4);
      if ( v51 == v49 - v51 )
      {
        v34 = *(_QWORD *)(a1 + 16);
        v35 = (int *)(a1 + 8);
        v36 = a2 + 2;
        if ( !v34 )
          goto LABEL_47;
LABEL_38:
        v37 = *v35;
        v53 = v34;
        v52 = v37;
        v54 = (int *)*((_QWORD *)v35 + 2);
        v55 = (int *)*((_QWORD *)v35 + 3);
        *(_QWORD *)(v34 + 8) = &v52;
        for ( i = *((_QWORD *)v35 + 4); ; i = 0 )
        {
          v38 = *((_QWORD *)v35 + 5);
          *((_QWORD *)v35 + 1) = 0;
          *((_QWORD *)v35 + 2) = v35;
          v57 = v38;
          v39 = *((_QWORD *)v35 + 6);
          *((_QWORD *)v35 + 3) = v35;
          v58 = v39;
          v40 = *((_QWORD *)v35 + 7);
          *((_QWORD *)v35 + 4) = 0;
          v12 = *((_QWORD *)v36 + 1) == 0;
          v59 = v40;
          v60 = *((_QWORD *)v35 + 8);
          if ( !v12 )
          {
            *v35 = *v36;
            v41 = *((_QWORD *)v36 + 1);
            *((_QWORD *)v35 + 1) = v41;
            *((_QWORD *)v35 + 2) = *((_QWORD *)v36 + 2);
            *((_QWORD *)v35 + 3) = *((_QWORD *)v36 + 3);
            *(_QWORD *)(v41 + 8) = v35;
            *((_QWORD *)v35 + 4) = *((_QWORD *)v36 + 4);
            *((_QWORD *)v36 + 1) = 0;
            *((_QWORD *)v36 + 2) = v36;
            *((_QWORD *)v36 + 3) = v36;
            *((_QWORD *)v36 + 4) = 0;
          }
          *((_QWORD *)v35 + 5) = *((_QWORD *)v36 + 5);
          *((_QWORD *)v35 + 6) = *((_QWORD *)v36 + 6);
          *((_QWORD *)v35 + 7) = *((_QWORD *)v36 + 7);
          *((_QWORD *)v35 + 8) = *((_QWORD *)v36 + 8);
          v42 = *((_QWORD *)v36 + 1);
          while ( v42 )
          {
            sub_1876060(*(_QWORD *)(v42 + 24));
            v43 = v42;
            v42 = *(_QWORD *)(v42 + 16);
            j_j___libc_free_0(v43, 40);
          }
          v44 = v53;
          *((_QWORD *)v36 + 1) = 0;
          *((_QWORD *)v36 + 2) = v36;
          *((_QWORD *)v36 + 3) = v36;
          *((_QWORD *)v36 + 4) = 0;
          if ( v44 )
          {
            v45 = v52;
            *((_QWORD *)v36 + 1) = v44;
            *v36 = v45;
            *((_QWORD *)v36 + 2) = v54;
            *((_QWORD *)v36 + 3) = v55;
            *(_QWORD *)(v44 + 8) = v36;
            v53 = 0;
            *((_QWORD *)v36 + 4) = i;
            v54 = &v52;
            v55 = &v52;
            i = 0;
          }
          v36 += 20;
          *((_QWORD *)v36 - 5) = v57;
          *((_QWORD *)v36 - 4) = v58;
          *((_QWORD *)v36 - 3) = v59;
          *((_QWORD *)v36 - 2) = v60;
          sub_1876060(0);
          v46 = v35 + 20;
          if ( a2 == v35 + 18 )
            break;
          v35 += 20;
          v34 = *((_QWORD *)v46 + 1);
          if ( v34 )
            goto LABEL_38;
LABEL_47:
          v52 = 0;
          v53 = 0;
          v54 = &v52;
          v55 = &v52;
        }
        return a2;
      }
      else
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v50 = v49 - v51;
            if ( v51 >= v49 - v51 )
              break;
            if ( v49 - v51 > 0 )
            {
              v4 = 0;
              v5 = (int *)(v48 + 8);
              v6 = v48 + 80 * v51 + 8;
              do
              {
                v7 = *((_QWORD *)v5 + 1);
                if ( v7 )
                {
                  v8 = *v5;
                  v53 = *((_QWORD *)v5 + 1);
                  v52 = v8;
                  v54 = (int *)*((_QWORD *)v5 + 2);
                  v55 = (int *)*((_QWORD *)v5 + 3);
                  *(_QWORD *)(v7 + 8) = &v52;
                  i = *((_QWORD *)v5 + 4);
                }
                else
                {
                  v52 = 0;
                  v53 = 0;
                  v54 = &v52;
                  v55 = &v52;
                  i = 0;
                }
                v9 = *((_QWORD *)v5 + 5);
                *((_QWORD *)v5 + 1) = 0;
                *((_QWORD *)v5 + 2) = v5;
                v57 = v9;
                v10 = *((_QWORD *)v5 + 6);
                *((_QWORD *)v5 + 3) = v5;
                v58 = v10;
                v11 = *((_QWORD *)v5 + 7);
                *((_QWORD *)v5 + 4) = 0;
                v12 = *(_QWORD *)(v6 + 8) == 0;
                v59 = v11;
                v60 = *((_QWORD *)v5 + 8);
                if ( !v12 )
                {
                  *v5 = *(_DWORD *)v6;
                  v13 = *(_QWORD *)(v6 + 8);
                  *((_QWORD *)v5 + 1) = v13;
                  *((_QWORD *)v5 + 2) = *(_QWORD *)(v6 + 16);
                  *((_QWORD *)v5 + 3) = *(_QWORD *)(v6 + 24);
                  *(_QWORD *)(v13 + 8) = v5;
                  *((_QWORD *)v5 + 4) = *(_QWORD *)(v6 + 32);
                  *(_QWORD *)(v6 + 8) = 0;
                  *(_QWORD *)(v6 + 16) = v6;
                  *(_QWORD *)(v6 + 24) = v6;
                  *(_QWORD *)(v6 + 32) = 0;
                }
                *((_QWORD *)v5 + 5) = *(_QWORD *)(v6 + 40);
                *((_QWORD *)v5 + 6) = *(_QWORD *)(v6 + 48);
                *((_QWORD *)v5 + 7) = *(_QWORD *)(v6 + 56);
                *((_QWORD *)v5 + 8) = *(_QWORD *)(v6 + 64);
                v14 = *(_QWORD *)(v6 + 8);
                while ( v14 )
                {
                  sub_1876060(*(_QWORD *)(v14 + 24));
                  v15 = v14;
                  v14 = *(_QWORD *)(v14 + 16);
                  j_j___libc_free_0(v15, 40);
                }
                v16 = v53;
                *(_QWORD *)(v6 + 8) = 0;
                *(_QWORD *)(v6 + 16) = v6;
                *(_QWORD *)(v6 + 24) = v6;
                *(_QWORD *)(v6 + 32) = 0;
                if ( v16 )
                {
                  v17 = v52;
                  *(_QWORD *)(v6 + 8) = v16;
                  *(_DWORD *)v6 = v17;
                  *(_QWORD *)(v6 + 16) = v54;
                  *(_QWORD *)(v6 + 24) = v55;
                  *(_QWORD *)(v16 + 8) = v6;
                  *(_QWORD *)(v6 + 32) = i;
                }
                ++v4;
                v5 += 20;
                v6 += 80;
                *(_QWORD *)(v6 - 40) = v57;
                *(_QWORD *)(v6 - 32) = v58;
                *(_QWORD *)(v6 - 24) = v59;
                *(_QWORD *)(v6 - 16) = v60;
              }
              while ( v50 != v4 );
              v48 += 80 * v50;
            }
            v18 = v49 % v51;
            if ( !(v49 % v51) )
              return (int *)v47;
            v49 = v51;
            v51 -= v18;
          }
          v19 = v48 + 80 * v49;
          v48 = v19 - 80 * v50;
          if ( v51 > 0 )
          {
            v20 = v19 - 80 * v50 - 72;
            v21 = v19 - 72;
            v22 = 0;
            do
            {
              v23 = *(_QWORD *)(v20 + 8);
              if ( v23 )
              {
                v24 = *(_DWORD *)v20;
                v53 = *(_QWORD *)(v20 + 8);
                v52 = v24;
                v54 = *(int **)(v20 + 16);
                v55 = *(int **)(v20 + 24);
                *(_QWORD *)(v23 + 8) = &v52;
                v25 = *(_QWORD *)(v20 + 32);
                *(_QWORD *)(v20 + 8) = 0;
                i = v25;
                *(_QWORD *)(v20 + 16) = v20;
                *(_QWORD *)(v20 + 24) = v20;
                *(_QWORD *)(v20 + 32) = 0;
              }
              else
              {
                v52 = 0;
                v53 = 0;
                v54 = &v52;
                v55 = &v52;
                i = 0;
              }
              v26 = *(_QWORD *)(v20 + 40);
              *(_QWORD *)(v20 + 8) = 0;
              *(_QWORD *)(v20 + 16) = v20;
              v57 = v26;
              v27 = *(_QWORD *)(v20 + 48);
              *(_QWORD *)(v20 + 24) = v20;
              v58 = v27;
              v28 = *(_QWORD *)(v20 + 56);
              *(_QWORD *)(v20 + 32) = 0;
              v12 = *(_QWORD *)(v21 + 8) == 0;
              v59 = v28;
              v60 = *(_QWORD *)(v20 + 64);
              if ( !v12 )
              {
                *(_DWORD *)v20 = *(_DWORD *)v21;
                v29 = *(_QWORD *)(v21 + 8);
                *(_QWORD *)(v20 + 8) = v29;
                *(_QWORD *)(v20 + 16) = *(_QWORD *)(v21 + 16);
                *(_QWORD *)(v20 + 24) = *(_QWORD *)(v21 + 24);
                *(_QWORD *)(v29 + 8) = v20;
                *(_QWORD *)(v20 + 32) = *(_QWORD *)(v21 + 32);
                *(_QWORD *)(v21 + 8) = 0;
                *(_QWORD *)(v21 + 16) = v21;
                *(_QWORD *)(v21 + 24) = v21;
                *(_QWORD *)(v21 + 32) = 0;
              }
              *(_QWORD *)(v20 + 40) = *(_QWORD *)(v21 + 40);
              *(_QWORD *)(v20 + 48) = *(_QWORD *)(v21 + 48);
              *(_QWORD *)(v20 + 56) = *(_QWORD *)(v21 + 56);
              *(_QWORD *)(v20 + 64) = *(_QWORD *)(v21 + 64);
              v30 = *(_QWORD *)(v21 + 8);
              while ( v30 )
              {
                sub_1876060(*(_QWORD *)(v30 + 24));
                v31 = v30;
                v30 = *(_QWORD *)(v30 + 16);
                j_j___libc_free_0(v31, 40);
              }
              v32 = v53;
              *(_QWORD *)(v21 + 8) = 0;
              *(_QWORD *)(v21 + 16) = v21;
              *(_QWORD *)(v21 + 24) = v21;
              *(_QWORD *)(v21 + 32) = 0;
              if ( v32 )
              {
                v33 = v52;
                *(_QWORD *)(v21 + 8) = v32;
                *(_DWORD *)v21 = v33;
                *(_QWORD *)(v21 + 16) = v54;
                *(_QWORD *)(v21 + 24) = v55;
                *(_QWORD *)(v32 + 8) = v21;
                *(_QWORD *)(v21 + 32) = i;
              }
              ++v22;
              v20 -= 80;
              v21 -= 80;
              *(_QWORD *)(v21 + 120) = v57;
              *(_QWORD *)(v21 + 128) = v58;
              *(_QWORD *)(v21 + 136) = v59;
              *(_QWORD *)(v21 + 144) = v60;
            }
            while ( v51 != v22 );
            v48 += -80 * v51;
          }
          v51 = v49 % v50;
          if ( !(v49 % v50) )
            break;
          v49 = v50;
        }
        return (int *)v47;
      }
    }
  }
  return result;
}
