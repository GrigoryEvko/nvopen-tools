// Function: sub_2D5AD50
// Address: 0x2d5ad50
//
__int64 (__fastcall *__fastcall sub_2D5AD50(
        char *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5))(int, int, int, int, int, int, __int64)
{
  unsigned __int8 **v8; // rdx
  unsigned __int8 *v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 (__fastcall *result)(int, int, int, int, int, int, __int64); // rax
  __int64 v13; // r15
  char v14; // r14
  __int64 v15; // rdi
  bool v16; // al
  __int64 v17; // r12
  __int64 (*v18)(); // rax
  __int64 v19; // rax
  unsigned int v20; // r12d
  bool v21; // al
  __int64 v22; // rax
  char *v23; // rdi
  __int64 v24; // rax
  char v25; // r9
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // rsi
  char **v29; // rcx
  char *v30; // r10
  __int64 v31; // rcx
  unsigned __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  _BYTE *v35; // rdi
  __int64 v36; // rax
  _BYTE *v37; // rcx
  unsigned int v38; // eax
  __int64 v39; // rax
  int v40; // ecx
  int v41; // r10d
  __int64 v42; // rcx
  char v43; // [rsp+8h] [rbp-48h]
  unsigned __int8 v44; // [rsp+10h] [rbp-40h]
  __int64 v45; // [rsp+10h] [rbp-40h]
  int v46; // [rsp+10h] [rbp-40h]
  __int64 v47; // [rsp+18h] [rbp-38h]
  __int64 v48; // [rsp+18h] [rbp-38h]
  unsigned int v49; // [rsp+18h] [rbp-38h]
  __int64 v50; // [rsp+18h] [rbp-38h]
  __int64 v51; // [rsp+18h] [rbp-38h]

  if ( (a1[7] & 0x40) != 0 )
    v8 = (unsigned __int8 **)*((_QWORD *)a1 - 1);
  else
    v8 = (unsigned __int8 **)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v9 = *v8;
  v10 = **v8;
  if ( (unsigned __int8)v10 <= 0x1Cu )
    return 0;
  v11 = *((_QWORD *)v9 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
    return 0;
  v13 = *((_QWORD *)a1 + 1);
  v14 = *a1;
  result = sub_2D63820;
  if ( (_BYTE)v10 != 68 )
  {
    LOBYTE(a5) = v14 == 69;
    if ( (_BYTE)v10 != 69 || v14 != 69 )
    {
      if ( (unsigned int)(unsigned __int8)v10 - 42 <= 0x11 && (unsigned __int8)v10 <= 0x36u )
      {
        v15 = 0x40540000000000LL;
        if ( _bittest64(&v15, (unsigned __int8)v10) )
        {
          v44 = v10;
          v47 = *((_QWORD *)v9 + 1);
          v16 = v14 == 69 ? sub_B44900((__int64)v9) : sub_B448F0((__int64)v9);
          v11 = v47;
          LODWORD(v10) = v44;
          a5 = v14 == 69;
          if ( v16 )
          {
LABEL_17:
            v10 = (unsigned int)(v10 - 67);
            result = sub_2D63820;
            if ( (unsigned __int8)v10 <= 2u )
              return result;
            goto LABEL_18;
          }
        }
LABEL_35:
        if ( (_BYTE)v10 == 54 )
        {
          v17 = *((_QWORD *)v9 + 2);
          if ( v17 )
          {
            if ( !*(_QWORD *)(v17 + 8) )
            {
              v34 = *(_QWORD *)(*(_QWORD *)(v17 + 24) + 16LL);
              if ( v34 )
              {
                if ( !*(_QWORD *)(v34 + 8) )
                {
                  v35 = *(_BYTE **)(v34 + 24);
                  if ( *v35 == 57 )
                  {
                    v45 = v11;
                    v36 = sub_986520((__int64)v35);
                    v37 = *(_BYTE **)(v36 + 32);
                    if ( *v37 == 17 )
                    {
                      v50 = *(_QWORD *)(v36 + 32);
                      v38 = sub_9871A0((__int64)(v37 + 24));
                      v10 = v50;
                      a5 = v38;
                      if ( *(_DWORD *)(v45 + 8) >> 8 >= *(_DWORD *)(v50 + 32) - v38 )
                      {
LABEL_19:
                        if ( !v17 || *(_QWORD *)(v17 + 8) )
                        {
                          v18 = *(__int64 (**)())(*(_QWORD *)a3 + 1376LL);
                          if ( v18 == sub_2D56660
                            || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64))v18)(
                                  a3,
                                  v13,
                                  *((_QWORD *)v9 + 1),
                                  v10,
                                  a5) )
                          {
                            return 0;
                          }
                        }
                        result = sub_2D63810;
                        if ( v14 != 69 )
                          return sub_2D63800;
                        return result;
                      }
                    }
                  }
                }
              }
            }
          }
          return 0;
        }
        v43 = a5;
        v48 = v11;
        if ( (_BYTE)v10 != 67 )
          return 0;
        v22 = sub_986520((__int64)v9);
        v23 = *(char **)v22;
        v24 = *(_QWORD *)(*(_QWORD *)v22 + 8LL);
        if ( *(_BYTE *)(v24 + 8) != 12 )
          return 0;
        if ( *(_DWORD *)(v24 + 8) >> 8 > *(_DWORD *)(v13 + 8) >> 8 )
          return 0;
        v25 = *v23;
        if ( (unsigned __int8)*v23 <= 0x1Cu )
          return 0;
        v26 = v48;
        v27 = *(_QWORD *)(a4 + 8);
        v28 = *(unsigned int *)(a4 + 24);
        if ( (_DWORD)v28 )
        {
          v49 = (v28 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v29 = (char **)(v27 + 16LL * v49);
          v30 = *v29;
          if ( v23 == *v29 )
          {
LABEL_42:
            if ( v29 != (char **)(v27 + 16 * v28) )
            {
              v31 = (__int64)v29[1];
              if ( v43 == ((v31 >> 1) & 3) )
              {
                v32 = v31 & 0xFFFFFFFFFFFFFFF8LL;
                if ( v32 )
                {
LABEL_45:
                  v33 = *(_DWORD *)(v26 + 8) >> 8;
                  if ( (unsigned int)v33 >= *(_DWORD *)(v32 + 8) >> 8
                    && !(unsigned __int8)sub_B19060(a2, (__int64)v9, v33, v32) )
                  {
                    result = sub_2D63820;
                    v10 = (unsigned int)*v9 - 67;
                    if ( (unsigned __int8)(*v9 - 67) <= 2u )
                      return result;
                    goto LABEL_18;
                  }
                  return 0;
                }
              }
            }
          }
          else
          {
            v40 = 1;
            while ( v30 != (char *)-4096LL )
            {
              v41 = v40 + 1;
              v42 = ((_DWORD)v28 - 1) & (v49 + v40);
              v46 = v41;
              v49 = v42;
              v29 = (char **)(v27 + 16 * v42);
              v30 = *v29;
              if ( v23 == *v29 )
                goto LABEL_42;
              v40 = v46;
            }
          }
        }
        if ( v14 == 69 )
        {
          if ( v25 != 69 )
            return 0;
        }
        else if ( v25 != 68 )
        {
          return 0;
        }
        v51 = v26;
        v39 = sub_986520((__int64)v23);
        v26 = v51;
        v32 = *(_QWORD *)(*(_QWORD *)v39 + 8LL);
        goto LABEL_45;
      }
      if ( (unsigned __int8)(v10 - 57) <= 1u )
        goto LABEL_17;
      if ( (_BYTE)v10 == 59 )
      {
        v19 = *(_QWORD *)(sub_986520((__int64)v9) + 32);
        if ( *(_BYTE *)v19 != 17 )
          return 0;
        v20 = *(_DWORD *)(v19 + 32);
        if ( !v20 )
          return 0;
        if ( v20 <= 0x40 )
        {
          v10 = 64 - v20;
          v21 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v20) == *(_QWORD *)(v19 + 24);
        }
        else
        {
          v21 = v20 == (unsigned int)sub_C445E0(v19 + 24);
        }
        if ( v21 )
          return 0;
      }
      else if ( (_BYTE)v10 != 55 || v14 == 69 )
      {
        goto LABEL_35;
      }
LABEL_18:
      v17 = *((_QWORD *)v9 + 2);
      goto LABEL_19;
    }
  }
  return result;
}
