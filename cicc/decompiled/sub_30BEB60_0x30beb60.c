// Function: sub_30BEB60
// Address: 0x30beb60
//
void __fastcall sub_30BEB60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  char *v7; // r12
  char *v8; // r15
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // r14
  unsigned __int64 *v14; // r14
  unsigned __int64 *v15; // r15
  __int64 v16; // r12
  unsigned __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // r10
  int v20; // eax
  __int64 v21; // rsi
  int v22; // edx
  unsigned int v23; // eax
  __int64 *v24; // rdi
  __int64 v25; // r10
  __int64 v26; // rax
  char *v27; // rdi
  int v28; // edx
  char *v29; // rsi
  __int64 v30; // r10
  __int64 v31; // rax
  char *v32; // rax
  char *v33; // r10
  __int64 v34; // rdi
  void (__fastcall *v35)(__int64, unsigned __int64); // rax
  __int64 *v36; // rax
  __int64 v37; // rdi
  __int64 (__fastcall *v38)(__int64, __int64, __int64); // rax
  __int64 v39; // rax
  __int64 v40; // r10
  int v41; // edi
  int v42; // r8d
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 (__fastcall *v45)(__int64, __int64, __int64); // rax
  __int64 v46; // rax
  __int64 v47; // r10
  char *v48; // rax
  __int64 v49; // [rsp+0h] [rbp-D0h]
  __int64 v50; // [rsp+0h] [rbp-D0h]
  __int64 v51; // [rsp+0h] [rbp-D0h]
  __int64 v52; // [rsp+0h] [rbp-D0h]
  __int64 v53; // [rsp+0h] [rbp-D0h]
  unsigned __int64 *v56; // [rsp+20h] [rbp-B0h]
  __int64 v58; // [rsp+38h] [rbp-98h] BYREF
  unsigned __int64 *v59; // [rsp+40h] [rbp-90h] BYREF
  __int64 v60; // [rsp+48h] [rbp-88h]
  _BYTE v61[128]; // [rsp+50h] [rbp-80h] BYREF

  v7 = *(char **)(a2 + 40);
  v8 = &v7[8 * *(unsigned int *)(a2 + 48)];
  if ( v8 != sub_30B9560(v7, v8, a3) )
  {
    v59 = (unsigned __int64 *)v61;
    v60 = 0xA00000000LL;
    if ( v7 != v8 )
    {
      v11 = v10 + 8;
      v12 = 0;
      do
      {
        while ( 1 )
        {
          v13 = *(_QWORD **)v7;
          if ( v11 == **(_QWORD **)v7 + 8LL )
            break;
          v7 += 8;
          if ( v8 == v7 )
            goto LABEL_10;
        }
        if ( v12 + 1 > (unsigned __int64)HIDWORD(v60) )
        {
          v53 = v11;
          sub_C8D5F0((__int64)&v59, v61, v12 + 1, 8u, v9, v10);
          v12 = (unsigned int)v60;
          v11 = v53;
        }
        v7 += 8;
        v59[v12] = (unsigned __int64)v13;
        v12 = (unsigned int)(v60 + 1);
        LODWORD(v60) = v60 + 1;
      }
      while ( v8 != v7 );
LABEL_10:
      v14 = &v59[v12];
      if ( v59 != v14 )
      {
        v56 = &v59[v12];
        v15 = v59;
        v16 = 4LL * a5;
        while ( 1 )
        {
          v17 = *v15;
          v18 = *a1;
          v19 = *(int *)(*v15 + 8);
          if ( *(_BYTE *)(*a1 + v16 + v19) )
            goto LABEL_16;
          if ( a5 )
          {
            if ( a5 == 1 )
            {
              v43 = (__int64 *)a1[1];
              switch ( (_DWORD)v19 )
              {
                case 2:
                  v44 = *v43;
                  v50 = *(int *)(*v15 + 8);
                  v45 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)*v43 + 48LL);
                  if ( v45 != sub_30B2FE0 )
                    goto LABEL_77;
                  v46 = sub_22077B0(0x10u);
                  v47 = v50;
                  if ( v46 )
                  {
                    *(_DWORD *)(v46 + 8) = 2;
                    *(_QWORD *)v46 = a3;
                  }
                  break;
                case 3:
                  v44 = *v43;
                  v50 = *(int *)(*v15 + 8);
                  v45 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)*v43 + 56LL);
                  if ( v45 != sub_30B2F90 )
                    goto LABEL_77;
                  v46 = sub_22077B0(0x10u);
                  v47 = v50;
                  if ( v46 )
                  {
                    *(_DWORD *)(v46 + 8) = 3;
                    *(_QWORD *)v46 = a3;
                  }
                  break;
                case 1:
                  v44 = *v43;
                  v50 = *(int *)(*v15 + 8);
                  v45 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)*v43 + 40LL);
                  if ( v45 != sub_30B3030 )
                  {
LABEL_77:
                    v45(v44, a4, a3);
                    v18 = *a1;
                    v19 = v50;
                    goto LABEL_15;
                  }
                  v46 = sub_22077B0(0x10u);
                  v47 = v50;
                  if ( v46 )
                  {
                    *(_DWORD *)(v46 + 8) = 1;
                    *(_QWORD *)v46 = a3;
                  }
                  break;
                default:
LABEL_82:
                  BUG();
              }
              v58 = v46;
              v51 = v47;
              sub_30B2AC0(a4 + 8, &v58);
              v18 = *a1;
              v19 = v51;
            }
          }
          else
          {
            v36 = (__int64 *)a1[1];
            switch ( (_DWORD)v19 )
            {
              case 2:
                v37 = *v36;
                v49 = *(int *)(*v15 + 8);
                v38 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)*v36 + 48LL);
                if ( v38 != sub_30B2FE0 )
                  goto LABEL_66;
                v39 = sub_22077B0(0x10u);
                v40 = v49;
                if ( v39 )
                {
                  *(_DWORD *)(v39 + 8) = 2;
                  *(_QWORD *)v39 = a4;
                }
                break;
              case 3:
                v37 = *v36;
                v49 = *(int *)(*v15 + 8);
                v38 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)*v36 + 56LL);
                if ( v38 != sub_30B2F90 )
                  goto LABEL_66;
                v39 = sub_22077B0(0x10u);
                v40 = v49;
                if ( v39 )
                {
                  *(_DWORD *)(v39 + 8) = 3;
                  *(_QWORD *)v39 = a4;
                }
                break;
              case 1:
                v37 = *v36;
                v49 = *(int *)(*v15 + 8);
                v38 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)*v36 + 40LL);
                if ( v38 != sub_30B3030 )
                {
LABEL_66:
                  v38(v37, a2, a4);
                  v18 = *a1;
                  v19 = v49;
                  goto LABEL_15;
                }
                v39 = sub_22077B0(0x10u);
                v40 = v49;
                if ( v39 )
                {
                  *(_DWORD *)(v39 + 8) = 1;
                  *(_QWORD *)v39 = a4;
                }
                break;
              default:
                goto LABEL_82;
            }
            v52 = v40;
            v58 = v39;
            sub_30B2AC0(a2 + 8, &v58);
            v18 = *a1;
            v19 = v52;
          }
LABEL_15:
          *(_BYTE *)(v16 + v18 + v19) = 1;
LABEL_16:
          v20 = *(_DWORD *)(a2 + 32);
          v21 = *(_QWORD *)(a2 + 16);
          if ( v20 )
          {
            v22 = v20 - 1;
            v23 = (v20 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v24 = (__int64 *)(v21 + 8LL * v23);
            v25 = *v24;
            if ( v17 != *v24 )
            {
              v41 = 1;
              while ( v25 != -4096 )
              {
                v42 = v41 + 1;
                v23 = v22 & (v41 + v23);
                v24 = (__int64 *)(v21 + 8LL * v23);
                v25 = *v24;
                if ( v17 == *v24 )
                  goto LABEL_18;
                v41 = v42;
              }
              goto LABEL_29;
            }
LABEL_18:
            *v24 = -8192;
            v26 = *(unsigned int *)(a2 + 48);
            v27 = *(char **)(a2 + 40);
            --*(_DWORD *)(a2 + 24);
            v28 = v26;
            v26 *= 8;
            ++*(_DWORD *)(a2 + 28);
            v29 = &v27[v26];
            v30 = v26 >> 3;
            v31 = v26 >> 5;
            if ( v31 )
            {
              v32 = &v27[32 * v31];
              while ( v17 != *(_QWORD *)v27 )
              {
                if ( v17 == *((_QWORD *)v27 + 1) )
                {
                  v27 += 8;
                  v33 = v27 + 8;
                  goto LABEL_26;
                }
                if ( v17 == *((_QWORD *)v27 + 2) )
                {
                  v27 += 16;
                  v33 = v27 + 8;
                  goto LABEL_26;
                }
                if ( v17 == *((_QWORD *)v27 + 3) )
                {
                  v27 += 24;
                  v33 = v27 + 8;
                  goto LABEL_26;
                }
                v27 += 32;
                if ( v32 == v27 )
                {
                  v30 = (v29 - v27) >> 3;
                  goto LABEL_68;
                }
              }
LABEL_25:
              v33 = v27 + 8;
LABEL_26:
              if ( v33 != v29 )
              {
                memmove(v27, v33, v29 - v33);
                v28 = *(_DWORD *)(a2 + 48);
              }
              *(_DWORD *)(a2 + 48) = v28 - 1;
              goto LABEL_29;
            }
LABEL_68:
            switch ( v30 )
            {
              case 2LL:
                v48 = v27;
                break;
              case 3LL:
                v33 = v27 + 8;
                v48 = v27 + 8;
                if ( v17 == *(_QWORD *)v27 )
                  goto LABEL_26;
                break;
              case 1LL:
LABEL_75:
                if ( v17 == *(_QWORD *)v27 )
                  goto LABEL_25;
                goto LABEL_71;
              default:
LABEL_71:
                v27 = v29;
                v33 = v29 + 8;
                goto LABEL_26;
            }
            v27 = v48 + 8;
            if ( v17 == *(_QWORD *)v48 )
            {
              v27 = v48;
              goto LABEL_25;
            }
            goto LABEL_75;
          }
LABEL_29:
          v34 = a1[2];
          v35 = *(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v34 + 72LL);
          if ( v35 == sub_30B0290 )
            j_j___libc_free_0(v17);
          else
            v35(v34, v17);
          if ( v56 == ++v15 )
          {
            v14 = v59;
            break;
          }
        }
      }
      if ( v14 != (unsigned __int64 *)v61 )
        _libc_free((unsigned __int64)v14);
    }
  }
}
