// Function: sub_38CE540
// Address: 0x38ce540
//
__int64 __fastcall sub_38CE540(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 *a5,
        _QWORD *a6,
        __int64 *a7)
{
  __int64 result; // rax
  __int64 v9; // r14
  __int64 *v10; // r11
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 *v18; // r11
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // edi
  int v26; // edi
  __int64 v27; // r8
  unsigned int v28; // esi
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rdx
  unsigned int v32; // r11d
  __int64 *v33; // rsi
  __int64 v34; // r9
  __int64 v35; // rdi
  unsigned __int64 v36; // rax
  int v37; // esi
  int v38; // edx
  int v39; // r10d
  int v40; // r11d
  __int64 v41; // [rsp+8h] [rbp-58h]
  bool v42; // [rsp+17h] [rbp-49h]
  __int64 *v43; // [rsp+18h] [rbp-48h]
  __int64 *v44; // [rsp+18h] [rbp-48h]
  __int64 *v45; // [rsp+18h] [rbp-48h]
  __int64 *v46; // [rsp+18h] [rbp-48h]
  __int64 *v47; // [rsp+18h] [rbp-48h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  __int64 *v49; // [rsp+18h] [rbp-48h]

  result = *a5;
  if ( *a5 )
  {
    if ( *a6 )
    {
      v9 = *(_QWORD *)(result + 24);
      v10 = *(__int64 **)(*a6 + 24LL);
      if ( (*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_7;
      result = *(_BYTE *)(v9 + 9) & 0xC;
      if ( (_BYTE)result == 8 )
      {
        *(_BYTE *)(v9 + 8) |= 4u;
        v43 = v10;
        v14 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
        v10 = v43;
        v15 = v14;
        result = v14 | *(_QWORD *)v9 & 7LL;
        *(_QWORD *)v9 = result;
        if ( v15 )
        {
LABEL_7:
          if ( (*v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            goto LABEL_59;
          result = *((_BYTE *)v10 + 9) & 0xC;
          if ( (_BYTE)result == 8 )
          {
            *((_BYTE *)v10 + 8) |= 4u;
            v44 = v10;
            v16 = (unsigned __int64)sub_38CE440(v10[3]);
            v10 = v44;
            v17 = v16;
            result = v16 | *v44 & 7;
            *v44 = result;
            if ( v17 )
            {
LABEL_59:
              v45 = v10;
              result = sub_38D6D30(*(_QWORD *)(a1 + 24), a1, *a5, *a6, a4);
              if ( (_BYTE)result )
              {
                v18 = v45;
                v19 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v19 )
                {
                  v19 = 0;
                  if ( (*(_BYTE *)(v9 + 9) & 0xC) == 8 )
                  {
                    *(_BYTE *)(v9 + 8) |= 4u;
                    v36 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
                    v18 = v45;
                    v19 = v36;
                    *(_QWORD *)v9 = v36 | *(_QWORD *)v9 & 7LL;
                  }
                }
                result = *v18 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !result )
                {
                  result = 0;
                  if ( (*((_BYTE *)v18 + 9) & 0xC) == 8 )
                  {
                    *((_BYTE *)v18 + 8) |= 4u;
                    v49 = v18;
                    result = (__int64)sub_38CE440(v18[3]);
                    v18 = v49;
                    *v49 = result | *v49 & 7;
                  }
                }
                if ( result == v19 && (*(_BYTE *)(v9 + 9) & 4) != 0 && (*((_BYTE *)v18 + 9) & 4) != 0 )
                {
                  *a7 += *(_QWORD *)(v9 + 24) - v18[3];
                  if ( (unsigned __int8)sub_390AF00(a1, v9) )
                    *a7 |= 1uLL;
                  v35 = *(_QWORD *)(a1 + 8);
                  result = *(_QWORD *)(*(_QWORD *)v35 + 152LL);
                  if ( (__int64 (*)())result == sub_38CB1D0 )
                    goto LABEL_37;
                  result = ((__int64 (__fastcall *)(__int64, __int64))result)(v35, v9);
                  if ( !(_BYTE)result )
                    goto LABEL_37;
                  goto LABEL_36;
                }
                if ( a2 )
                {
                  v20 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( !v20 )
                  {
                    v46 = v18;
                    if ( (*(_BYTE *)(v9 + 9) & 0xC) != 8 )
                      BUG();
                    *(_BYTE *)(v9 + 8) |= 4u;
                    v20 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
                    v18 = v46;
                    *(_QWORD *)v9 = v20 | *(_QWORD *)v9 & 7LL;
                  }
                  v21 = *(_QWORD *)(v20 + 24);
                  result = *v18 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( !result )
                  {
                    if ( (*((_BYTE *)v18 + 9) & 0xC) != 8 )
                      BUG();
                    *((_BYTE *)v18 + 8) |= 4u;
                    v47 = v18;
                    result = (__int64)sub_38CE440(v18[3]);
                    *v47 = result | *v47 & 7;
                  }
                  v22 = *(_QWORD *)(result + 24);
                  if ( a3 || v22 == v21 )
                  {
                    v41 = *(_QWORD *)(result + 24);
                    v42 = v22 != v21;
                    v48 = sub_38D0440(a2, *(_QWORD *)(*a5 + 24));
                    v23 = v48 + *a7 - sub_38D0440(a2, *(_QWORD *)(*a6 + 24LL));
                    *a7 = v23;
                    v24 = v23;
                    if ( a3 && v42 )
                    {
                      v25 = *(_DWORD *)(a3 + 24);
                      if ( v25 )
                      {
                        v26 = v25 - 1;
                        v27 = *(_QWORD *)(a3 + 8);
                        v28 = v26 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                        v29 = (__int64 *)(v27 + 16LL * v28);
                        v30 = *v29;
                        if ( v21 == *v29 )
                        {
LABEL_31:
                          v31 = v29[1];
                        }
                        else
                        {
                          v38 = 1;
                          while ( v30 != -8 )
                          {
                            v40 = v38 + 1;
                            v28 = v26 & (v38 + v28);
                            v29 = (__int64 *)(v27 + 16LL * v28);
                            v30 = *v29;
                            if ( v21 == *v29 )
                              goto LABEL_31;
                            v38 = v40;
                          }
                          v31 = 0;
                        }
                        v32 = v26 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                        v33 = (__int64 *)(v27 + 16LL * v32);
                        v34 = *v33;
                        if ( v41 == *v33 )
                        {
LABEL_33:
                          v24 = v23 + v31 - v33[1];
                        }
                        else
                        {
                          v37 = 1;
                          while ( v34 != -8 )
                          {
                            v39 = v37 + 1;
                            v32 = v26 & (v37 + v32);
                            v33 = (__int64 *)(v27 + 16LL * v32);
                            v34 = *v33;
                            if ( v41 == *v33 )
                              goto LABEL_33;
                            v37 = v39;
                          }
                          v24 = v23 + v31;
                        }
                      }
                      *a7 = v24;
                    }
                    result = sub_390AF00(a1, v9);
                    if ( !(_BYTE)result )
                      goto LABEL_37;
LABEL_36:
                    result = (__int64)a7;
                    *a7 |= 1uLL;
LABEL_37:
                    *a6 = 0;
                    *a5 = 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
