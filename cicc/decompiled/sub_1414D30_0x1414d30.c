// Function: sub_1414D30
// Address: 0x1414d30
//
void __fastcall sub_1414D30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rdx
  int v6; // ebx
  unsigned __int64 *v7; // r9
  int v8; // r14d
  unsigned __int64 *v9; // rdi
  unsigned __int64 v10; // rbx
  __int64 *v11; // r15
  unsigned __int64 v12; // rdx
  __int64 *v13; // rax
  char v14; // cl
  __int64 *v15; // rdx
  unsigned __int64 v16; // r15
  _BOOL4 v17; // r9d
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rbx
  __int64 v21; // rax
  unsigned __int8 v22; // dl
  unsigned __int64 *v23; // rsi
  __int64 *v24; // rsi
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rax
  __int64 *v28; // r12
  __int64 *v29; // rbx
  _QWORD *v30; // r8
  _QWORD *v31; // r9
  __int64 v32; // rsi
  _QWORD *v33; // rdi
  unsigned int v34; // r10d
  _QWORD *v35; // rax
  _QWORD *v36; // rcx
  _BOOL4 v38; // [rsp+2Ch] [rbp-D4h]
  __int64 *v39; // [rsp+30h] [rbp-D0h]
  unsigned __int8 v40; // [rsp+30h] [rbp-D0h]
  unsigned __int64 *v41; // [rsp+30h] [rbp-D0h]
  __int64 v42; // [rsp+38h] [rbp-C8h]
  __int64 v44; // [rsp+58h] [rbp-A8h] BYREF
  void *dest; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 *v46; // [rsp+68h] [rbp-98h]
  unsigned __int64 *v47; // [rsp+70h] [rbp-90h]
  __int64 *v48; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v49; // [rsp+88h] [rbp-78h]
  __int64 *v50; // [rsp+90h] [rbp-70h]
  __int64 v51; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v52; // [rsp+A8h] [rbp-58h] BYREF
  __int64 *v53; // [rsp+B0h] [rbp-50h]
  __int64 *v54; // [rsp+B8h] [rbp-48h]
  __int64 *v55; // [rsp+C0h] [rbp-40h]
  __int64 v56; // [rsp+C8h] [rbp-38h]

  if ( byte_4F99740 )
  {
    v2 = sub_1649960(a2);
    v52 = v3;
    v51 = v2;
    if ( sub_16D20C0(&v51, "cutlass", 7, 0) == -1 )
    {
      if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      {
        sub_15E08E0(a2);
        v4 = *(_QWORD *)(a2 + 88);
        if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
          sub_15E08E0(a2);
        v5 = *(_QWORD *)(a2 + 88);
      }
      else
      {
        v4 = *(_QWORD *)(a2 + 88);
        v5 = v4;
      }
      v42 = v5 + 40LL * *(_QWORD *)(a2 + 96);
      while ( v4 != v42 )
      {
        v6 = sub_15E0450(v4);
        if ( (_BYTE)v6 )
        {
          dest = 0;
          v54 = &v52;
          v55 = &v52;
          v48 = 0;
          v46 = 0;
          v47 = 0;
          v49 = 0;
          v50 = 0;
          LODWORD(v52) = 0;
          v53 = 0;
          v56 = 0;
          v44 = v4;
          sub_12879C0((__int64)&dest, 0, &v44);
LABEL_11:
          v7 = v46;
          v8 = v6;
LABEL_12:
          v9 = (unsigned __int64 *)dest;
          if ( dest != v7 )
          {
            do
            {
              v10 = *v9;
              if ( v9 + 1 != v7 )
              {
                memmove(v9, v9 + 1, (char *)v7 - (char *)(v9 + 1));
                v7 = v46;
              }
              v11 = v53;
              v46 = --v7;
              if ( v53 )
              {
                while ( 1 )
                {
                  v12 = v11[4];
                  v13 = (__int64 *)v11[3];
                  v14 = 0;
                  if ( v10 < v12 )
                  {
                    v13 = (__int64 *)v11[2];
                    v14 = v8;
                  }
                  if ( !v13 )
                    break;
                  v11 = v13;
                }
                if ( !v14 )
                {
                  if ( v12 >= v10 )
                    goto LABEL_12;
LABEL_23:
                  v15 = v11;
                  v16 = v10;
                  v6 = v8;
LABEL_24:
                  v17 = 1;
                  if ( v15 != &v52 )
                    v17 = v16 < v15[4];
                  v38 = v17;
                  v39 = v15;
                  v18 = sub_22077B0(40);
                  *(_QWORD *)(v18 + 32) = v16;
                  sub_220F040(v38, v18, v39, &v52);
                  ++v56;
                  v19 = *(_QWORD *)(v16 + 8);
                  if ( !v19 )
                    goto LABEL_11;
                  v40 = v6;
                  v20 = v19;
                  while ( 1 )
                  {
                    v21 = sub_1648700(v20);
                    v22 = *(_BYTE *)(v21 + 16);
                    if ( v22 <= 0x17u )
                      goto LABEL_28;
                    if ( (unsigned __int8)(v22 - 71) > 1u && v22 != 56 )
                    {
                      if ( v22 != 78 )
                      {
                        if ( v22 == 54 )
                        {
                          v44 = v21;
                          v24 = v49;
                          if ( v49 == v50 )
                          {
                            sub_14147F0((__int64)&v48, v49, &v44);
                          }
                          else
                          {
                            if ( v49 )
                            {
                              *v49 = v21;
                              v24 = v49;
                            }
                            v49 = v24 + 1;
                          }
                        }
                        else
                        {
LABEL_28:
                          v44 = 0;
                          if ( !(unsigned __int8)sub_1C2FEA0(a2, (unsigned int)(*(_DWORD *)(v4 + 32) + 1)) )
                            goto LABEL_49;
                        }
LABEL_29:
                        v20 = *(_QWORD *)(v20 + 8);
                        if ( !v20 )
                          goto LABEL_37;
                        continue;
                      }
                      v25 = *(_QWORD *)(v21 - 24);
                      if ( *(_BYTE *)(v25 + 16) || (*(_BYTE *)(v25 + 33) & 0x20) == 0 )
                        goto LABEL_28;
                      v26 = *(_DWORD *)(v25 + 36);
                      if ( v26 != 4046 && v26 != 4242 )
                        goto LABEL_49;
                    }
                    v44 = v21;
                    v23 = v46;
                    if ( v46 == v47 )
                    {
                      sub_12879C0((__int64)&dest, v46, &v44);
                      goto LABEL_29;
                    }
                    if ( v46 )
                    {
                      *v46 = v21;
                      v23 = v46;
                    }
                    v46 = v23 + 1;
                    v20 = *(_QWORD *)(v20 + 8);
                    if ( !v20 )
                    {
LABEL_37:
                      v6 = v40;
                      goto LABEL_11;
                    }
                  }
                }
                if ( v11 == v54 )
                  goto LABEL_23;
              }
              else
              {
                if ( v54 == &v52 )
                {
                  v16 = v10;
                  v15 = &v52;
                  v6 = v8;
                  goto LABEL_24;
                }
                v11 = &v52;
              }
              v41 = v7;
              v27 = sub_220EF80(v11);
              v7 = v41;
              if ( *(_QWORD *)(v27 + 32) < v10 )
                goto LABEL_23;
              v9 = (unsigned __int64 *)dest;
            }
            while ( dest != v41 );
          }
          v28 = v48;
          v29 = v49;
          if ( v48 != v49 )
          {
            v30 = *(_QWORD **)(a1 + 16);
            v31 = *(_QWORD **)(a1 + 8);
            do
            {
              v32 = *v28;
              if ( v30 != v31 )
                goto LABEL_58;
              v33 = &v30[*(unsigned int *)(a1 + 28)];
              v34 = *(_DWORD *)(a1 + 28);
              if ( v33 != v30 )
              {
                v35 = v30;
                v36 = 0;
                while ( v32 != *v35 )
                {
                  if ( *v35 == -2 )
                    v36 = v35;
                  if ( v33 == ++v35 )
                  {
                    if ( !v36 )
                      goto LABEL_76;
                    *v36 = v32;
                    v30 = *(_QWORD **)(a1 + 16);
                    --*(_DWORD *)(a1 + 32);
                    v31 = *(_QWORD **)(a1 + 8);
                    ++*(_QWORD *)a1;
                    goto LABEL_59;
                  }
                }
                goto LABEL_59;
              }
LABEL_76:
              if ( v34 < *(_DWORD *)(a1 + 24) )
              {
                *(_DWORD *)(a1 + 28) = v34 + 1;
                *v33 = v32;
                v31 = *(_QWORD **)(a1 + 8);
                ++*(_QWORD *)a1;
                v30 = *(_QWORD **)(a1 + 16);
              }
              else
              {
LABEL_58:
                sub_16CCBA0(a1, v32);
                v30 = *(_QWORD **)(a1 + 16);
                v31 = *(_QWORD **)(a1 + 8);
              }
LABEL_59:
              ++v28;
            }
            while ( v29 != v28 );
          }
LABEL_49:
          sub_1411CA0((__int64)v53);
          if ( v48 )
            j_j___libc_free_0(v48, (char *)v50 - (char *)v48);
          if ( dest )
            j_j___libc_free_0(dest, (char *)v47 - (_BYTE *)dest);
        }
        v4 += 40;
      }
    }
  }
}
