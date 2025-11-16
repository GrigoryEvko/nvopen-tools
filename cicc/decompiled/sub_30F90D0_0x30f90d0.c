// Function: sub_30F90D0
// Address: 0x30f90d0
//
__int64 __fastcall sub_30F90D0(__int64 a1, __int64 a2)
{
  unsigned int v4; // eax
  unsigned int v5; // r14d
  __int64 *v7; // rax
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned int v11; // esi
  __int64 v12; // rdi
  int v13; // r11d
  __int64 *v14; // r9
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // r15
  __int64 v21; // rdi
  int v22; // eax
  int v23; // edx
  unsigned int v24; // eax
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  unsigned int v28; // ecx
  __int64 v29; // rdi
  int v30; // r8d
  __int64 *v31; // r10
  int v32; // eax
  int v33; // ecx
  __int64 v34; // rsi
  __int64 *v35; // r8
  unsigned int v36; // r15d
  int v37; // edi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r13
  unsigned int v41; // esi
  __int64 v42; // r8
  int v43; // edi
  __int64 *v44; // rcx
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // r10
  __int64 *v48; // rax
  bool v49; // cc
  int v50; // eax
  __int64 v51; // rdx
  __int64 v52; // [rsp+0h] [rbp-80h] BYREF
  __int64 *v53; // [rsp+8h] [rbp-78h] BYREF
  unsigned __int64 v54; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v55; // [rsp+18h] [rbp-68h]
  char v56; // [rsp+20h] [rbp-60h]
  __int64 v57; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v58; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v59; // [rsp+40h] [rbp-40h]

  LOBYTE(v4) = sub_D97040(*(_QWORD *)(a1 + 48), *(_QWORD *)(a2 + 8));
  v5 = v4;
  if ( (_BYTE)v4 )
  {
    v7 = sub_DD8400(*(_QWORD *)(a1 + 48), a2);
    v8 = (__int64)v7;
    if ( !*((_WORD *)v7 + 12) )
    {
      v9 = *(_QWORD *)(a1 + 40);
      v10 = v7[4];
      v11 = *(_DWORD *)(v9 + 24);
      if ( v11 )
      {
        v12 = *(_QWORD *)(v9 + 8);
        v13 = 1;
        v14 = 0;
        v15 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v16 = (__int64 *)(v12 + 16LL * v15);
        v17 = *v16;
        if ( a2 == *v16 )
        {
LABEL_6:
          v18 = v16 + 1;
LABEL_7:
          *v18 = v10;
          return v5;
        }
        while ( v17 != -4096 )
        {
          if ( !v14 && v17 == -8192 )
            v14 = v16;
          v15 = (v11 - 1) & (v13 + v15);
          v16 = (__int64 *)(v12 + 16LL * v15);
          v17 = *v16;
          if ( a2 == *v16 )
            goto LABEL_6;
          ++v13;
        }
        if ( !v14 )
          v14 = v16;
        v22 = *(_DWORD *)(v9 + 16);
        ++*(_QWORD *)v9;
        v23 = v22 + 1;
        if ( 4 * (v22 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(v9 + 20) - v23 > v11 >> 3 )
          {
LABEL_24:
            *(_DWORD *)(v9 + 16) = v23;
            if ( *v14 != -4096 )
              --*(_DWORD *)(v9 + 20);
            *v14 = a2;
            v18 = v14 + 1;
            v14[1] = 0;
            goto LABEL_7;
          }
          sub_FAA400(v9, v11);
          v32 = *(_DWORD *)(v9 + 24);
          if ( v32 )
          {
            v33 = v32 - 1;
            v34 = *(_QWORD *)(v9 + 8);
            v35 = 0;
            v36 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v37 = 1;
            v23 = *(_DWORD *)(v9 + 16) + 1;
            v14 = (__int64 *)(v34 + 16LL * v36);
            v38 = *v14;
            if ( a2 != *v14 )
            {
              while ( v38 != -4096 )
              {
                if ( !v35 && v38 == -8192 )
                  v35 = v14;
                v36 = v33 & (v37 + v36);
                v14 = (__int64 *)(v34 + 16LL * v36);
                v38 = *v14;
                if ( a2 == *v14 )
                  goto LABEL_24;
                ++v37;
              }
              if ( v35 )
                v14 = v35;
            }
            goto LABEL_24;
          }
LABEL_87:
          ++*(_DWORD *)(v9 + 16);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)v9;
      }
      sub_FAA400(v9, 2 * v11);
      v25 = *(_DWORD *)(v9 + 24);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(v9 + 8);
        v28 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v23 = *(_DWORD *)(v9 + 16) + 1;
        v14 = (__int64 *)(v27 + 16LL * v28);
        v29 = *v14;
        if ( a2 != *v14 )
        {
          v30 = 1;
          v31 = 0;
          while ( v29 != -4096 )
          {
            if ( !v31 && v29 == -8192 )
              v31 = v14;
            v28 = v26 & (v30 + v28);
            v14 = (__int64 *)(v27 + 16LL * v28);
            v29 = *v14;
            if ( a2 == *v14 )
              goto LABEL_24;
            ++v30;
          }
          if ( v31 )
            v14 = v31;
        }
        goto LABEL_24;
      }
      goto LABEL_87;
    }
    if ( !sub_D968A0(*(_QWORD *)(a1 + 32)) )
    {
      LOBYTE(v24) = sub_DADE90(*(_QWORD *)(a1 + 48), v8, *(_QWORD *)(a1 + 56));
      if ( (_BYTE)v24 )
        return v24;
    }
    if ( *(_WORD *)(v8 + 24) != 8 || *(_QWORD *)(v8 + 48) != *(_QWORD *)(a1 + 56) )
      return 0;
    v19 = sub_DD0540(v8, *(_QWORD *)(a1 + 32), *(__int64 **)(a1 + 48));
    v20 = (__int64)v19;
    if ( !*((_WORD *)v19 + 12) )
    {
      v21 = *(_QWORD *)(a1 + 40);
      v10 = v19[4];
      v57 = a2;
      v18 = sub_FAA780(v21, &v57);
      goto LABEL_7;
    }
    v39 = sub_D97190(*(_QWORD *)(a1 + 48), v8);
    v40 = v39;
    if ( *(_WORD *)(v39 + 24) != 15 )
      return 0;
    sub_DC06D0((__int64)&v54, *(_QWORD *)(a1 + 48), v20, v39);
    if ( !v56 )
      return 0;
    v59 = 1;
    v58 = 0;
    v57 = *(_QWORD *)(v40 - 8);
    if ( v55 <= 0x40 )
    {
      v59 = v55;
      v58 = v54;
    }
    else
    {
      sub_C43990((__int64)&v58, (__int64)&v54);
    }
    v41 = *(_DWORD *)(a1 + 24);
    v52 = a2;
    if ( v41 )
    {
      v42 = *(_QWORD *)(a1 + 8);
      v43 = 1;
      v44 = 0;
      v45 = (v41 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v46 = (__int64 *)(v42 + 32LL * v45);
      v47 = *v46;
      if ( a2 == *v46 )
      {
LABEL_49:
        v48 = v46 + 1;
        v49 = *((_DWORD *)v48 + 4) <= 0x40u;
        *v48 = v57;
        if ( v49 && v59 <= 0x40 )
        {
          v48[1] = v58;
          *((_DWORD *)v48 + 4) = v59;
        }
        else
        {
          sub_C43990((__int64)(v48 + 1), (__int64)&v58);
        }
        if ( v59 > 0x40 && v58 )
          j_j___libc_free_0_0(v58);
        if ( v56 )
        {
          v56 = 0;
          if ( v55 > 0x40 )
          {
            if ( v54 )
              j_j___libc_free_0_0(v54);
          }
        }
        return 0;
      }
      while ( v47 != -4096 )
      {
        if ( v47 == -8192 && !v44 )
          v44 = v46;
        v45 = (v41 - 1) & (v43 + v45);
        v46 = (__int64 *)(v42 + 32LL * v45);
        v47 = *v46;
        if ( a2 == *v46 )
          goto LABEL_49;
        ++v43;
      }
      if ( v44 )
        v46 = v44;
      ++*(_QWORD *)a1;
      v53 = v46;
      v50 = *(_DWORD *)(a1 + 16) + 1;
      if ( 4 * v50 < 3 * v41 )
      {
        if ( v41 - *(_DWORD *)(a1 + 20) - v50 > v41 >> 3 )
        {
LABEL_81:
          v46 = v53;
          ++*(_DWORD *)(a1 + 16);
          if ( *v46 != -4096 )
            --*(_DWORD *)(a1 + 20);
          v51 = v52;
          v46[1] = 0;
          *((_DWORD *)v46 + 6) = 1;
          *v46 = v51;
          v46[2] = 0;
          goto LABEL_49;
        }
LABEL_86:
        sub_30F8EB0(a1, v41);
        sub_30F8CD0(a1, &v52, &v53);
        goto LABEL_81;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v53 = 0;
    }
    v41 *= 2;
    goto LABEL_86;
  }
  return v5;
}
