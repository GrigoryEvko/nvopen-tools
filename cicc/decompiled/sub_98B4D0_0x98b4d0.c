// Function: sub_98B4D0
// Address: 0x98b4d0
//
__int64 __fastcall sub_98B4D0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  _QWORD *v6; // rdi
  __int64 result; // rax
  char *v8; // rsi
  unsigned __int8 *v9; // rdi
  unsigned __int8 *v10; // rbx
  unsigned __int8 **v11; // rax
  unsigned __int8 **v12; // rdx
  char v13; // dl
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // r8
  __int64 v24; // rsi
  int v25; // edx
  unsigned int v26; // edi
  __int64 *v27; // rcx
  __int64 v28; // r10
  __int64 *v29; // r9
  __int64 v30; // r9
  __int64 v31; // rax
  unsigned __int8 *v32; // rcx
  char *v33; // rdx
  __int64 v34; // r10
  __int64 v35; // r8
  __int64 v36; // rdi
  __int64 v37; // r11
  __int64 *v38; // rcx
  __int64 v39; // r12
  __int64 *v40; // r9
  __int64 v41; // rdx
  __int64 v42; // r11
  unsigned int v43; // r12d
  int v44; // r9d
  int v45; // ecx
  int v46; // r9d
  int v47; // ecx
  int v48; // r9d
  __int64 v49; // rcx
  int v50; // r9d
  int v51; // ecx
  __int64 v52; // r9
  __int64 v53; // [rsp+10h] [rbp-C0h]
  int v54; // [rsp+1Ch] [rbp-B4h]
  unsigned int v55; // [rsp+1Ch] [rbp-B4h]
  char *v56; // [rsp+20h] [rbp-B0h]
  _QWORD *v58; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v59; // [rsp+38h] [rbp-98h]
  _QWORD v60[4]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v61; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int8 **v62; // [rsp+68h] [rbp-68h]
  __int64 v63; // [rsp+70h] [rbp-60h]
  int v64; // [rsp+78h] [rbp-58h]
  char v65; // [rsp+7Ch] [rbp-54h]
  char v66; // [rsp+80h] [rbp-50h] BYREF

  v61 = 0;
  v63 = 4;
  v64 = 0;
  v65 = 1;
  v58 = v60;
  v56 = (char *)(a2 + 16);
  v62 = (unsigned __int8 **)&v66;
  v60[0] = a1;
  v6 = v60;
  v59 = 0x400000001LL;
  LODWORD(result) = 1;
  do
  {
    v8 = (char *)a4;
    v9 = (unsigned __int8 *)v6[(unsigned int)result - 1];
    LODWORD(v59) = result - 1;
    v10 = sub_98ACB0(v9, a4);
    if ( !v65 )
      goto LABEL_13;
    v11 = v62;
    v12 = &v62[HIDWORD(v63)];
    if ( v62 != v12 )
    {
      while ( v10 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_20;
      }
      goto LABEL_7;
    }
LABEL_20:
    if ( HIDWORD(v63) < (unsigned int)v63 )
    {
      ++HIDWORD(v63);
      *v12 = v10;
      ++v61;
    }
    else
    {
LABEL_13:
      v8 = (char *)v10;
      sub_C8CC70(&v61, v10);
      if ( !v13 )
        goto LABEL_7;
    }
    v14 = *v10;
    if ( *v10 <= 0x1Cu )
      goto LABEL_17;
    if ( v14 != 86 )
    {
      if ( v14 != 84 )
        goto LABEL_17;
      v21 = *((_DWORD *)v10 + 1) & 0x7FFFFFF;
      if ( a3 )
      {
        v22 = *(_DWORD *)(a3 + 24);
        v23 = *((_QWORD *)v10 + 5);
        v24 = *(_QWORD *)(a3 + 8);
        if ( v22 )
        {
          v25 = v22 - 1;
          v26 = v25 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v27 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v27;
          v29 = v27;
          if ( v23 == *v27 )
          {
LABEL_30:
            v30 = v29[1];
            if ( v30 && v23 == **(_QWORD **)(v30 + 32) )
            {
              if ( v23 == v28 )
              {
LABEL_38:
                v34 = v27[1];
              }
              else
              {
                v45 = 1;
                while ( v28 != -4096 )
                {
                  v48 = v45 + 1;
                  v49 = v25 & (v26 + v45);
                  v26 = v49;
                  v27 = (__int64 *)(v24 + 16 * v49);
                  v28 = *v27;
                  if ( v23 == *v27 )
                    goto LABEL_38;
                  v45 = v48;
                }
                v34 = 0;
              }
              if ( (_DWORD)v21 == 2 )
              {
                v35 = **((_QWORD **)v10 - 1);
                if ( *(_BYTE *)v35 > 0x1Cu )
                {
                  v36 = *(_QWORD *)(v35 + 40);
                  LODWORD(v37) = v25 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
                  v38 = (__int64 *)(v24 + 16LL * (unsigned int)v37);
                  v39 = *v38;
                  v40 = v38;
                  if ( v36 != *v38 )
                  {
                    v53 = *v38;
                    v46 = 1;
                    v55 = v25 & (((unsigned int)*(_QWORD *)(v35 + 40) >> 9) ^ ((unsigned int)v36 >> 4));
                    while ( v53 != -4096 )
                    {
                      v51 = v46 + 1;
                      v52 = v25 & (v55 + v46);
                      v55 = v52;
                      v40 = (__int64 *)(v24 + 16 * v52);
                      v53 = *v40;
                      if ( v36 == *v40 )
                      {
                        v38 = (__int64 *)(v24
                                        + 16LL
                                        * (v25 & (((unsigned int)*(_QWORD *)(v35 + 40) >> 9) ^ ((unsigned int)v36 >> 4))));
                        goto LABEL_42;
                      }
                      v46 = v51;
                    }
                    if ( v34 )
                      goto LABEL_43;
                    goto LABEL_62;
                  }
LABEL_42:
                  if ( v34 == v40[1] )
                  {
LABEL_46:
                    if ( v39 == v36 )
                    {
LABEL_47:
                      v41 = v38[1];
                      goto LABEL_48;
                    }
LABEL_62:
                    v47 = 1;
                    while ( v39 != -4096 )
                    {
                      v50 = v47 + 1;
                      LODWORD(v37) = v25 & (v47 + v37);
                      v38 = (__int64 *)(v24 + 16LL * (unsigned int)v37);
                      v39 = *v38;
                      if ( v36 == *v38 )
                        goto LABEL_47;
                      v47 = v50;
                    }
                    v41 = 0;
LABEL_48:
                    if ( v34 == v41 && *(_BYTE *)v35 == 61 )
                    {
                      v8 = *(char **)(v35 - 32);
                      if ( !(unsigned __int8)sub_D48480(v34, v8) )
                      {
LABEL_17:
                        v15 = *(unsigned int *)(a2 + 8);
                        if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
                        {
                          v8 = v56;
                          sub_C8D5F0(a2, v56, v15 + 1, 8);
                          v15 = *(unsigned int *)(a2 + 8);
                        }
                        *(_QWORD *)(*(_QWORD *)a2 + 8 * v15) = v10;
                        ++*(_DWORD *)(a2 + 8);
LABEL_7:
                        result = (unsigned int)v59;
                        goto LABEL_8;
                      }
                      v21 = *((_DWORD *)v10 + 1) & 0x7FFFFFF;
                    }
                    goto LABEL_32;
                  }
                }
LABEL_43:
                v35 = *(_QWORD *)(*((_QWORD *)v10 - 1) + 32LL);
                if ( !v35 )
                  BUG();
                if ( *(_BYTE *)v35 > 0x1Cu )
                {
                  v36 = *(_QWORD *)(v35 + 40);
                  v37 = v25 & (((unsigned int)v36 >> 4) ^ ((unsigned int)v36 >> 9));
                  v38 = (__int64 *)(v24 + 16 * v37);
                  v39 = *v38;
                  goto LABEL_46;
                }
              }
            }
          }
          else
          {
            v42 = *v27;
            v43 = v25 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v44 = 1;
            while ( v42 != -4096 )
            {
              v43 = v25 & (v44 + v43);
              v54 = v44 + 1;
              v29 = (__int64 *)(v24 + 16LL * v43);
              v42 = *v29;
              if ( v23 == *v29 )
                goto LABEL_30;
              v44 = v54;
            }
          }
        }
      }
LABEL_32:
      v31 = 32 * v21;
      if ( (v10[7] & 0x40) != 0 )
      {
        v33 = (char *)*((_QWORD *)v10 - 1);
        v32 = (unsigned __int8 *)&v33[v31];
      }
      else
      {
        v32 = v10;
        v33 = (char *)&v10[-v31];
      }
      v8 = (char *)&v58[(unsigned int)v59];
      sub_984620((__int64 *)&v58, v8, v33, v32);
      result = (unsigned int)v59;
      goto LABEL_8;
    }
    v16 = (unsigned int)v59;
    v17 = *((_QWORD *)v10 - 8);
    v18 = (unsigned int)v59 + 1LL;
    if ( v18 > HIDWORD(v59) )
    {
      v8 = (char *)v60;
      sub_C8D5F0(&v58, v60, v18, 8);
      v16 = (unsigned int)v59;
    }
    v58[v16] = v17;
    LODWORD(v59) = v59 + 1;
    v19 = (unsigned int)v59;
    v20 = *((_QWORD *)v10 - 4);
    if ( (unsigned __int64)(unsigned int)v59 + 1 > HIDWORD(v59) )
    {
      v8 = (char *)v60;
      sub_C8D5F0(&v58, v60, (unsigned int)v59 + 1LL, 8);
      v19 = (unsigned int)v59;
    }
    v58[v19] = v20;
    result = (unsigned int)(v59 + 1);
    LODWORD(v59) = v59 + 1;
LABEL_8:
    v6 = v58;
  }
  while ( (_DWORD)result );
  if ( v58 != v60 )
    result = _libc_free(v58, v8);
  if ( !v65 )
    return _libc_free(v62, v8);
  return result;
}
