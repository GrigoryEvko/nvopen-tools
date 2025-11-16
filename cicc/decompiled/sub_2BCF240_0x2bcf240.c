// Function: sub_2BCF240
// Address: 0x2bcf240
//
unsigned __int64 __fastcall sub_2BCF240(__int64 a1, unsigned __int8 *a2, __int64 a3, __m128i a4)
{
  int v5; // eax
  bool v6; // dl
  unsigned __int64 result; // rax
  unsigned __int8 *v9; // rcx
  unsigned __int8 *v10; // r14
  unsigned __int8 *v11; // r15
  __int64 v12; // rbx
  int v13; // eax
  __int64 v14; // rsi
  int v15; // edx
  unsigned int v16; // eax
  unsigned __int8 *v17; // rdi
  int v18; // r8d
  __int64 *v19; // rax
  __int64 v20; // r9
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rsi
  unsigned __int8 *v27; // rdx
  int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // rdi
  int v32; // edx
  int v33; // edx
  unsigned __int8 *v34; // rdx
  int v35; // eax
  __int64 *v36; // rax
  __int64 v37; // rcx
  __int64 *v38; // rax
  __int64 v39; // rcx
  __int64 *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // [rsp-C8h] [rbp-C8h]
  __int64 v43; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 v44; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 *v45; // [rsp-B8h] [rbp-B8h] BYREF
  unsigned __int8 *v46; // [rsp-B0h] [rbp-B0h] BYREF
  __int64 v47; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned __int8 *v48; // [rsp-A0h] [rbp-A0h] BYREF
  unsigned __int8 *v49; // [rsp-98h] [rbp-98h] BYREF
  unsigned __int8 *v50; // [rsp-90h] [rbp-90h]
  _QWORD *v51; // [rsp-88h] [rbp-88h] BYREF
  int v52; // [rsp-80h] [rbp-80h]
  int v53; // [rsp-7Ch] [rbp-7Ch]
  _QWORD v54[15]; // [rsp-78h] [rbp-78h] BYREF

  if ( !a2 )
    return 0;
  v5 = *a2;
  v6 = (unsigned __int8)(v5 - 82) <= 1u;
  result = (unsigned int)(v5 - 42);
  LOBYTE(result) = v6 || (unsigned int)result <= 0x11;
  if ( (_BYTE)result )
  {
    result = 0;
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a2 + 1) + 8LL) - 17 > 1 )
    {
      if ( (a2[7] & 0x40) != 0 )
      {
        v9 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        v10 = *(unsigned __int8 **)v9;
        if ( **(_BYTE **)v9 <= 0x1Cu )
          return result;
      }
      else
      {
        v9 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v10 = *(unsigned __int8 **)v9;
        if ( **(_BYTE **)v9 <= 0x1Cu )
          return result;
      }
      v11 = (unsigned __int8 *)*((_QWORD *)v9 + 4);
      if ( *v11 > 0x1Cu )
      {
        v12 = *((_QWORD *)a2 + 5);
        if ( v12 == *((_QWORD *)v10 + 5) && v12 == *((_QWORD *)v11 + 5) )
        {
          v13 = *(_DWORD *)(a3 + 2000);
          v14 = *(_QWORD *)(a3 + 1984);
          if ( v13 )
          {
            v15 = v13 - 1;
            v16 = (v13 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v17 = *(unsigned __int8 **)(v14 + 8LL * v16);
            if ( v10 == v17 )
              return 0;
            v18 = 1;
            while ( v17 != (unsigned __int8 *)-4096LL )
            {
              v16 = v15 & (v18 + v16);
              v17 = *(unsigned __int8 **)(v14 + 8LL * v16);
              if ( v10 == v17 )
                return 0;
              ++v18;
            }
          }
          v51 = (_QWORD *)*((_QWORD *)v9 + 4);
          v19 = sub_2B4B3F0(a3 + 1976, (__int64 *)&v51);
          v20 = a3 + 1976;
          if ( !v19 )
          {
            v53 = 4;
            v51 = v54;
            v54[0] = v10;
            v54[1] = v11;
            v52 = 1;
            if ( (unsigned int)*v10 - 42 > 0x11 )
            {
              v45 = 0;
              if ( (unsigned int)*v11 - 42 <= 0x11 )
              {
                v46 = v11;
                goto LABEL_22;
              }
            }
            else
            {
              v45 = v10;
              if ( (unsigned int)*v11 - 42 <= 0x11 )
              {
                v21 = *((_QWORD *)v11 + 2);
                v46 = v11;
                v22 = (__int64)v10;
                if ( !v21 || *(_QWORD *)(v21 + 8) )
                {
LABEL_21:
                  v23 = *(_QWORD *)(v22 + 16);
                  if ( v23 && !*(_QWORD *)(v23 + 8) )
                  {
                    v26 = *(_QWORD *)(v22 - 64);
                    if ( (unsigned __int8)(*(_BYTE *)v26 - 42) > 0x11u )
                    {
                      v47 = 0;
                      v34 = *(unsigned __int8 **)(v22 - 32);
                      v35 = *v34;
                      if ( (unsigned __int8)v35 <= 0x1Cu )
                      {
                        v48 = 0;
                        goto LABEL_22;
                      }
                      if ( (unsigned int)(v35 - 42) >= 0x12 )
                        v34 = 0;
                      v48 = v34;
                    }
                    else
                    {
                      v47 = *(_QWORD *)(v22 - 64);
                      v27 = *(unsigned __int8 **)(v22 - 32);
                      v28 = *v27;
                      if ( (unsigned __int8)v28 > 0x1Cu )
                      {
                        if ( (unsigned int)(v28 - 42) >= 0x12 )
                          v27 = 0;
                        v48 = v27;
                      }
                      else
                      {
                        v48 = 0;
                      }
                      if ( v12 == *(_QWORD *)(v26 + 40) )
                      {
                        v49 = (unsigned __int8 *)v26;
                        v42 = v20;
                        v36 = sub_2B4B3F0(v20, (__int64 *)&v49);
                        v20 = v42;
                        if ( !v36 )
                        {
                          sub_2B10C50((__int64)&v51, &v47, (__int64 *)&v46, v37, (__int64)&v51, v42);
                          v20 = v42;
                        }
                      }
                    }
                    if ( v48 )
                    {
                      if ( v12 == *((_QWORD *)v48 + 5) )
                      {
                        v49 = v48;
                        if ( !sub_2B4B3F0(v20, (__int64 *)&v49) )
                          sub_2B10C50((__int64)&v51, (__int64 *)&v48, (__int64 *)&v46, v29, (__int64)&v51, v30);
                      }
                    }
                  }
LABEL_22:
                  if ( v52 != 1 )
                  {
                    v24 = sub_2B65E20((_QWORD *)a3, (__int64)v51, v52, 0);
                    result = HIDWORD(v24);
                    v48 = (unsigned __int8 *)v24;
                    if ( !BYTE4(v24) )
                      goto LABEL_26;
                    v25 = &v51[2 * (int)v24];
                    v49 = (unsigned __int8 *)*v25;
                    v50 = (unsigned __int8 *)v25[1];
LABEL_25:
                    result = sub_2BCE070(a1, (__int64 *)&v49, 2u, a3, 0, a4);
LABEL_26:
                    if ( v51 != v54 )
                    {
                      v44 = result;
                      _libc_free((unsigned __int64)v51);
                      return v44;
                    }
                    return result;
                  }
LABEL_30:
                  v49 = v10;
                  v50 = v11;
                  goto LABEL_25;
                }
                v31 = *((_QWORD *)v11 - 8);
                if ( (unsigned __int8)(*(_BYTE *)v31 - 42) > 0x11u )
                {
                  v47 = 0;
                  v33 = **((unsigned __int8 **)v11 - 4);
                  if ( (unsigned __int8)v33 <= 0x1Cu )
                  {
                    v48 = 0;
                    goto LABEL_49;
                  }
                  if ( (unsigned int)(v33 - 42) < 0x12 )
                    v19 = (__int64 *)*((_QWORD *)v11 - 4);
                  v48 = (unsigned __int8 *)v19;
                }
                else
                {
                  v47 = *((_QWORD *)v11 - 8);
                  v32 = **((unsigned __int8 **)v11 - 4);
                  if ( (unsigned __int8)v32 > 0x1Cu )
                  {
                    if ( (unsigned int)(v32 - 42) < 0x12 )
                      v19 = (__int64 *)*((_QWORD *)v11 - 4);
                    v48 = (unsigned __int8 *)v19;
                  }
                  else
                  {
                    v48 = 0;
                  }
                  if ( v12 == *(_QWORD *)(v31 + 40) )
                  {
                    v49 = (unsigned __int8 *)v31;
                    v38 = sub_2B4B3F0(v20, (__int64 *)&v49);
                    v20 = a3 + 1976;
                    if ( !v38 )
                    {
                      sub_2B10C50((__int64)&v51, (__int64 *)&v45, &v47, v39, (__int64)&v51, v20);
                      v20 = a3 + 1976;
                    }
                  }
                }
                if ( v48 )
                {
                  if ( v12 == *((_QWORD *)v48 + 5) )
                  {
                    v43 = v20;
                    v49 = v48;
                    v40 = sub_2B4B3F0(v20, (__int64 *)&v49);
                    v20 = v43;
                    if ( !v40 )
                    {
                      sub_2B10C50((__int64)&v51, (__int64 *)&v45, (__int64 *)&v48, v41, (__int64)&v51, v43);
                      v20 = v43;
                    }
                  }
                }
LABEL_49:
                if ( !v46 )
                  goto LABEL_22;
                v22 = (__int64)v45;
                if ( !v45 )
                  goto LABEL_22;
                goto LABEL_21;
              }
            }
            v46 = 0;
            goto LABEL_30;
          }
        }
      }
      return 0;
    }
  }
  return result;
}
