// Function: sub_264C960
// Address: 0x264c960
//
__int64 *__fastcall sub_264C960(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *result; // rax
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // r8
  int v8; // r11d
  __int64 *v9; // r10
  unsigned int v10; // edx
  __int64 *v11; // rcx
  __int64 v12; // rdi
  int v13; // edx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  int v18; // edx
  unsigned int v19; // ecx
  int v20; // esi
  __int64 v21; // rdi
  __int64 v22; // rax
  int *v23; // rbx
  int *v24; // r12
  int *v25; // rax
  int *v26; // rbx
  int *v27; // r12
  int v28; // edx
  __int64 v29; // rdi
  __int64 v30; // rax
  unsigned int v31; // esi
  int *v32; // rcx
  int v33; // r9d
  int *v34; // rax
  int *v35; // r15
  int *v36; // r13
  int *v37; // rax
  int v38; // ecx
  int v39; // r8d
  __int64 v40; // r12
  int *v41; // r15
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v48; // [rsp+18h] [rbp-D8h]
  __int64 *v49; // [rsp+20h] [rbp-D0h]
  __int64 v51; // [rsp+38h] [rbp-B8h]
  __int64 *v52; // [rsp+40h] [rbp-B0h]
  __int64 v53; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v54; // [rsp+58h] [rbp-98h]
  int *v55; // [rsp+60h] [rbp-90h]
  int *v56; // [rsp+68h] [rbp-88h]
  __int64 v57; // [rsp+70h] [rbp-80h] BYREF
  __int64 v58; // [rsp+78h] [rbp-78h]
  __int64 v59; // [rsp+80h] [rbp-70h]
  __int64 v60; // [rsp+88h] [rbp-68h]
  __int64 *v61; // [rsp+90h] [rbp-60h] BYREF
  __int64 v62; // [rsp+98h] [rbp-58h]
  int v63; // [rsp+A0h] [rbp-50h]
  int v64; // [rsp+A4h] [rbp-4Ch]
  unsigned int v65; // [rsp+A8h] [rbp-48h]

  result = *(__int64 **)(a2 + 72);
  v49 = *(__int64 **)(a2 + 80);
  v52 = result;
  if ( result != v49 )
  {
    while ( 1 )
    {
      v5 = *v52;
      v6 = *(_DWORD *)(a3 + 24);
      v57 = *v52;
      if ( !v6 )
        break;
      v7 = *(_QWORD *)(a3 + 8);
      v8 = 1;
      v9 = 0;
      v10 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v11 = (__int64 *)(v7 + 8LL * v10);
      v12 = *v11;
      if ( v5 != *v11 )
      {
        while ( v12 != -4096 )
        {
          if ( !v9 && v12 == -8192 )
            v9 = v11;
          v10 = (v6 - 1) & (v8 + v10);
          v11 = (__int64 *)(v7 + 8LL * v10);
          v12 = *v11;
          if ( v5 == *v11 )
            goto LABEL_4;
          ++v8;
        }
        if ( v9 )
          v11 = v9;
        v13 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v61 = v11;
        v14 = v13 + 1;
        if ( 4 * v14 < 3 * v6 )
        {
          if ( v6 - *(_DWORD *)(a3 + 20) - v14 <= v6 >> 3 )
          {
            sub_2646080(a3, v6);
            sub_263DBF0(a3, &v57, &v61);
            v5 = v57;
            v11 = v61;
            v14 = *(_DWORD *)(a3 + 16) + 1;
          }
          goto LABEL_17;
        }
LABEL_56:
        sub_2646080(a3, 2 * v6);
        sub_263DBF0(a3, &v57, &v61);
        v5 = v57;
        v11 = v61;
        v14 = *(_DWORD *)(a3 + 16) + 1;
LABEL_17:
        *(_DWORD *)(a3 + 16) = v14;
        if ( *v11 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v11 = v5;
        v15 = *v52;
        v16 = *(_QWORD *)(*v52 + 8);
        v57 = 0;
        v58 = 0;
        v48 = v16;
        v17 = *a1;
        v59 = 0;
        v60 = 0;
        v18 = *(_DWORD *)(v15 + 40);
        v51 = v17;
        if ( v18 )
        {
          v23 = *(int **)(v15 + 32);
          v24 = &v23[*(unsigned int *)(v15 + 48)];
          if ( v23 == v24 )
            goto LABEL_27;
          while ( (unsigned int)*v23 > 0xFFFFFFFD )
          {
            if ( v24 == ++v23 )
              goto LABEL_27;
          }
          if ( v24 == v23 )
          {
LABEL_27:
            v19 = 0;
            v20 = 0;
            v18 = 0;
            v21 = 0;
            v22 = 1;
          }
          else
          {
            v25 = v23;
            v26 = v24;
            v27 = v25;
            do
            {
              v28 = *v27;
              v29 = *(_QWORD *)(*(_QWORD *)v51 + 8LL);
              v30 = *(unsigned int *)(*(_QWORD *)v51 + 24LL);
              if ( (_DWORD)v30 )
              {
                v31 = (v30 - 1) & (37 * v28);
                v32 = (int *)(v29 + 40LL * v31);
                v33 = *v32;
                if ( v28 == *v32 )
                {
LABEL_32:
                  if ( v32 != (int *)(v29 + 40 * v30) )
                  {
                    v34 = (int *)*((_QWORD *)v32 + 2);
                    if ( v32[6] )
                    {
                      v35 = &v34[v32[8]];
                      if ( v34 != v35 )
                      {
                        while ( 1 )
                        {
                          v36 = v34;
                          if ( (unsigned int)*v34 <= 0xFFFFFFFD )
                            break;
                          if ( v35 == ++v34 )
                            goto LABEL_34;
                        }
                        if ( v35 != v34 )
                        {
                          do
                          {
                            sub_22B6470((__int64)&v61, (__int64)&v57, v36);
                            v37 = v36 + 1;
                            if ( v35 == v36 + 1 )
                              break;
                            while ( 1 )
                            {
                              v36 = v37;
                              if ( (unsigned int)*v37 <= 0xFFFFFFFD )
                                break;
                              if ( v35 == ++v37 )
                                goto LABEL_34;
                            }
                          }
                          while ( v35 != v37 );
                        }
                      }
                    }
                  }
                }
                else
                {
                  v38 = 1;
                  while ( v33 != -1 )
                  {
                    v39 = v38 + 1;
                    v31 = (v30 - 1) & (v38 + v31);
                    v32 = (int *)(v29 + 40LL * v31);
                    v33 = *v32;
                    if ( v28 == *v32 )
                      goto LABEL_32;
                    v38 = v39;
                  }
                }
              }
LABEL_34:
              if ( ++v27 == v26 )
                break;
              while ( (unsigned int)*v27 > 0xFFFFFFFD )
              {
                if ( v26 == ++v27 )
                  goto LABEL_37;
              }
            }
            while ( v26 != v27 );
LABEL_37:
            v21 = v58;
            v18 = v59;
            v20 = HIDWORD(v59);
            v19 = v60;
            v22 = v57 + 1;
          }
        }
        else
        {
          v19 = 0;
          v20 = 0;
          v21 = 0;
          v22 = 1;
        }
        v62 = v21;
        v57 = v22;
        v61 = (__int64 *)1;
        v63 = v18;
        v64 = v20;
        v65 = v19;
        v58 = 0;
        v59 = 0;
        LODWORD(v60) = 0;
        sub_2342640((__int64)&v57);
        if ( v63 )
        {
          v40 = *v52 + 24;
          v41 = (int *)(v62 + 4LL * v65);
          sub_22B0690(&v53, (__int64 *)&v61);
          sub_264C8F0(v40, v41, v42, v43, v44, v45, v53, v54, v55, v56);
          sub_264C960(a4, v48, a3, a4);
        }
        sub_2342640((__int64)&v61);
      }
LABEL_4:
      v52 += 2;
      result = v52;
      if ( v49 == v52 )
        return result;
    }
    v61 = 0;
    ++*(_QWORD *)a3;
    goto LABEL_56;
  }
  return result;
}
