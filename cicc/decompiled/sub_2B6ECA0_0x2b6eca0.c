// Function: sub_2B6ECA0
// Address: 0x2b6eca0
//
__int64 __fastcall sub_2B6ECA0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v5; // rbx
  unsigned int v7; // r8d
  __int64 v8; // rdi
  int v9; // r11d
  __int64 *v10; // r10
  unsigned int v11; // eax
  __int64 *v12; // rcx
  __int64 v13; // r9
  _QWORD *v14; // r15
  __int64 v15; // r14
  int v16; // r10d
  __int64 *v17; // r9
  unsigned int v18; // eax
  __int64 *v19; // rsi
  __int64 v20; // rcx
  _QWORD *v21; // rcx
  int v22; // eax
  __int64 v23; // rbx
  __int64 *v24; // r8
  unsigned __int8 *v25; // rax
  int v26; // edi
  unsigned __int8 *v27; // rdx
  int v28; // r10d
  __int64 v29; // rsi
  __int64 v30; // r9
  int v31; // esi
  int v32; // esi
  unsigned int v33; // edi
  unsigned __int8 *v34; // r10
  int v35; // r11d
  unsigned int v36; // edi
  unsigned __int8 *v37; // r10
  int v38; // r11d
  __int64 *v39; // r9
  __int64 v40; // rdx
  int v41; // esi
  int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // rdx
  unsigned int v45; // eax
  int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // rdx
  int v49; // eax
  int v50; // eax
  _QWORD *v51; // [rsp+0h] [rbp-60h]
  __int64 *v52; // [rsp+8h] [rbp-58h]
  __int64 v53; // [rsp+10h] [rbp-50h] BYREF
  __int64 v54; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v55[8]; // [rsp+20h] [rbp-40h] BYREF

  v53 = a2;
  v54 = a3;
  if ( a3 == a2 )
    return 1;
  if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)(a3 + 8) )
  {
    v5 = *a1;
    v3 = *(_DWORD *)(*a1 + 24);
    if ( v3 )
    {
      v7 = v3 - 1;
      v8 = *(_QWORD *)(v5 + 8);
      v9 = 1;
      v10 = 0;
      v11 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = (__int64 *)(v8 + 56LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
      {
LABEL_7:
        v14 = (_QWORD *)v12[1];
        v15 = *((unsigned int *)v12 + 4);
        goto LABEL_8;
      }
      while ( v13 != -4096 )
      {
        if ( !v10 && v13 == -8192 )
          v10 = v12;
        v11 = v7 & (v9 + v11);
        v12 = (__int64 *)(v8 + 56LL * v11);
        v13 = *v12;
        if ( a2 == *v12 )
          goto LABEL_7;
        ++v9;
      }
      if ( !v10 )
        v10 = v12;
      v55[0] = v10;
      v50 = *(_DWORD *)(v5 + 16);
      ++*(_QWORD *)v5;
      v42 = v50 + 1;
      if ( 4 * (v50 + 1) < 3 * v3 )
      {
        if ( v3 - *(_DWORD *)(v5 + 20) - v42 > v3 >> 3 )
          goto LABEL_38;
        v41 = v3;
LABEL_37:
        sub_2B5BE90(v5, v41);
        sub_2B41380(v5, &v53, v55);
        v42 = *(_DWORD *)(v5 + 16) + 1;
LABEL_38:
        *(_DWORD *)(v5 + 16) = v42;
        v43 = (__int64 *)v55[0];
        if ( *(_QWORD *)v55[0] != -4096 )
          --*(_DWORD *)(v5 + 20);
        v44 = v53;
        v14 = v43 + 3;
        v43[1] = (__int64)(v43 + 3);
        *v43 = v44;
        v43[2] = 0x400000000LL;
        v5 = *a1;
        v3 = *(_DWORD *)(*a1 + 24);
        if ( !v3 )
        {
          v55[0] = 0;
          v45 = 0;
          ++*(_QWORD *)v5;
          goto LABEL_42;
        }
        v8 = *(_QWORD *)(v5 + 8);
        a3 = v54;
        v7 = v3 - 1;
        v15 = 0;
LABEL_8:
        v16 = 1;
        v17 = 0;
        v18 = v7 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v19 = (__int64 *)(v8 + 56LL * v18);
        v20 = *v19;
        if ( a3 == *v19 )
        {
LABEL_9:
          v21 = (_QWORD *)v19[1];
          v22 = *((_DWORD *)v19 + 4);
LABEL_10:
          if ( (_DWORD)v15 != v22 )
            return 0;
          if ( (int)v15 > 0 )
          {
            v23 = 0;
            v24 = v55;
            do
            {
              v25 = (unsigned __int8 *)v14[v23];
              v26 = *v25;
              if ( (unsigned int)(v26 - 12) > 1 )
              {
                v27 = (unsigned __int8 *)v21[v23];
                v28 = *v27;
                if ( (unsigned int)(v28 - 12) > 1 )
                {
                  LOBYTE(v3) = (unsigned __int8)v26 <= 0x1Cu || (unsigned __int8)v28 <= 0x1Cu;
                  if ( (_BYTE)v3 )
                  {
                    if ( ((unsigned __int8)v28 > 0x15u || (unsigned __int8)v26 > 0x15u) && v26 != v28 )
                      return 0;
                  }
                  else
                  {
                    v29 = a1[2];
                    v30 = *(_QWORD *)(v29 + 1984);
                    v31 = *(_DWORD *)(v29 + 2000);
                    if ( v31 )
                    {
                      v32 = v31 - 1;
                      v33 = v32 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
                      v34 = *(unsigned __int8 **)(v30 + 8LL * v33);
                      if ( v25 == v34 )
                        return 0;
                      v35 = 1;
                      while ( v34 != (unsigned __int8 *)-4096LL )
                      {
                        v33 = v32 & (v35 + v33);
                        v34 = *(unsigned __int8 **)(v30 + 8LL * v33);
                        if ( v25 == v34 )
                          return 0;
                        ++v35;
                      }
                      v36 = v32 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
                      v37 = *(unsigned __int8 **)(v30 + 8LL * v36);
                      if ( v27 == v37 )
                        return 0;
                      v38 = 1;
                      while ( v37 != (unsigned __int8 *)-4096LL )
                      {
                        v36 = v32 & (v38 + v36);
                        v37 = *(unsigned __int8 **)(v30 + 8LL * v36);
                        if ( v27 == v37 )
                          return 0;
                        ++v38;
                      }
                    }
                    if ( *((_QWORD *)v25 + 5) != *((_QWORD *)v27 + 5) )
                      return 0;
                    v51 = v21;
                    v52 = v24;
                    v39 = *(__int64 **)(a1[1] + 16);
                    v55[1] = v21[v23];
                    v55[0] = v25;
                    if ( !sub_2B5F980(v24, 2u, v39) )
                      return v3;
                    v24 = v52;
                    v21 = v51;
                    if ( !v40 )
                      return v3;
                  }
                }
              }
              ++v23;
            }
            while ( v15 != v23 );
          }
          return 1;
        }
        while ( v20 != -4096 )
        {
          if ( !v17 && v20 == -8192 )
            v17 = v19;
          v18 = v7 & (v16 + v18);
          v19 = (__int64 *)(v8 + 56LL * v18);
          v20 = *v19;
          if ( a3 == *v19 )
            goto LABEL_9;
          ++v16;
        }
        if ( !v17 )
          v17 = v19;
        v55[0] = v17;
        v49 = *(_DWORD *)(v5 + 16);
        ++*(_QWORD *)v5;
        v46 = v49 + 1;
        if ( 4 * (v49 + 1) < 3 * v3 )
        {
          if ( v3 - *(_DWORD *)(v5 + 20) - v46 <= v3 >> 3 )
          {
            sub_2B5BE90(v5, v3);
            sub_2B41380(v5, &v54, v55);
            v46 = *(_DWORD *)(v5 + 16) + 1;
          }
          goto LABEL_43;
        }
        v45 = v3;
        v3 = v15;
LABEL_42:
        v15 = v3;
        sub_2B5BE90(v5, 2 * v45);
        sub_2B41380(v5, &v54, v55);
        v46 = *(_DWORD *)(v5 + 16) + 1;
LABEL_43:
        *(_DWORD *)(v5 + 16) = v46;
        v47 = (__int64 *)v55[0];
        if ( *(_QWORD *)v55[0] != -4096 )
          --*(_DWORD *)(v5 + 20);
        v48 = v54;
        v21 = v47 + 3;
        v47[1] = (__int64)(v47 + 3);
        *v47 = v48;
        v47[2] = 0x400000000LL;
        v22 = 0;
        goto LABEL_10;
      }
    }
    else
    {
      v55[0] = 0;
      ++*(_QWORD *)v5;
    }
    v41 = 2 * v3;
    goto LABEL_37;
  }
  return 0;
}
