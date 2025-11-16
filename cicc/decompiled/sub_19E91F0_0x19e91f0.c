// Function: sub_19E91F0
// Address: 0x19e91f0
//
__int64 **__fastcall sub_19E91F0(__int64 a1)
{
  __int64 **result; // rax
  __int64 *v2; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r14
  __int64 v8; // rax
  char v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  __int64 v15; // r9
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r11
  char *v19; // rdx
  char v20; // al
  char *v21; // rdx
  char v22; // al
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 *v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned int v29; // r8d
  __int64 *v30; // rdx
  __int64 v31; // r10
  unsigned int v32; // r14d
  __int64 v33; // rsi
  __int64 *v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r9
  unsigned int v37; // esi
  __int64 *v38; // rdx
  __int64 v39; // r10
  int v40; // eax
  int v41; // r8d
  int v42; // edx
  int v43; // r8d
  int v44; // edx
  __int64 v45; // [rsp+0h] [rbp-70h]
  __int64 v46; // [rsp+8h] [rbp-68h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  int v48; // [rsp+10h] [rbp-60h]
  __int64 *v49; // [rsp+28h] [rbp-48h] BYREF
  __int64 v50[8]; // [rsp+30h] [rbp-40h] BYREF

  result = &v49;
  v2 = *(__int64 **)a1;
  if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(a1 + 16);
      if ( *(_BYTE *)(*(_QWORD *)v4 + 16LL) != 77 || *(_QWORD *)v4 != *v2 && (v5 = sub_19E1D90(*v2), v6 != v5) )
      {
        v7 = v2[1];
        v8 = **(_QWORD **)(a1 + 32);
        v50[0] = v7;
        v50[1] = v8;
        v9 = sub_19E8F30(*(_QWORD *)(a1 + 24) + 2200LL, v50, &v49);
        if ( v9 )
        {
          v10 = *(_QWORD *)(a1 + 24);
          v11 = *v2;
          v12 = 0;
          v13 = *(_DWORD *)(v10 + 1496);
          if ( v13 )
          {
            v14 = v13 - 1;
            v15 = *(_QWORD *)(v10 + 1480);
            v16 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v17 = (__int64 *)(v15 + 16LL * v16);
            v18 = *v17;
            if ( v11 == *v17 )
            {
LABEL_8:
              v12 = v17[1];
            }
            else
            {
              v40 = 1;
              while ( v18 != -8 )
              {
                v41 = v40 + 1;
                v16 = v14 & (v40 + v16);
                v17 = (__int64 *)(v15 + 16LL * v16);
                v18 = *v17;
                if ( v11 == *v17 )
                  goto LABEL_8;
                v40 = v41;
              }
              v12 = 0;
            }
          }
          if ( *(_QWORD *)(v10 + 1432) != v12 )
            break;
        }
      }
LABEL_15:
      result = *(__int64 ***)a1;
      v2 = (__int64 *)(*(_QWORD *)a1 + 16LL);
      *(_QWORD *)a1 = v2;
      if ( *(__int64 **)(a1 + 8) == v2 )
        return result;
    }
    v19 = *(char **)(a1 + 40);
    v20 = *v19;
    if ( *v19 )
      v20 = *(_BYTE *)(v11 + 16) <= 0x10u;
    *v19 = v20;
    v21 = *(char **)(a1 + 48);
    v22 = *v21;
    if ( *v21 || (v23 = **(_QWORD **)(a1 + 32), v22 = v9, v7 == v23) )
    {
LABEL_13:
      *v21 = v22;
      result = (__int64 **)sub_19E1ED0(*(_QWORD *)(a1 + 24), (__int64 ***)*v2);
      if ( **(__int64 ****)(a1 + 16) != result )
        return result;
      goto LABEL_15;
    }
    v24 = *(_QWORD *)(a1 + 24);
    v25 = 0;
    v26 = *(_QWORD *)(v24 + 8);
    v27 = *(unsigned int *)(v26 + 48);
    if ( (_DWORD)v27 )
    {
      v28 = *(_QWORD *)(v26 + 32);
      v29 = (v27 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v30 = (__int64 *)(v28 + 16LL * v29);
      v31 = *v30;
      if ( v7 == *v30 )
      {
LABEL_20:
        if ( v30 != (__int64 *)(v28 + 16 * v27) )
        {
          v25 = (__int64 *)v30[1];
          goto LABEL_22;
        }
      }
      else
      {
        v44 = 1;
        while ( v31 != -8 )
        {
          v29 = (v27 - 1) & (v44 + v29);
          v48 = v44 + 1;
          v30 = (__int64 *)(v28 + 16LL * v29);
          v31 = *v30;
          if ( v7 == *v30 )
            goto LABEL_20;
          v44 = v48;
        }
      }
      v25 = 0;
    }
LABEL_22:
    v49 = v25;
    v32 = 0;
    v45 = v23;
    v46 = v24;
    v47 = v24 + 1400;
    if ( (unsigned __int8)sub_19E6B80(v24 + 1400, (__int64 *)&v49, v50) )
      v32 = *(_DWORD *)(v50[0] + 8);
    v33 = *(_QWORD *)(v46 + 8);
    v34 = 0;
    v35 = *(unsigned int *)(v33 + 48);
    if ( !(_DWORD)v35 )
      goto LABEL_28;
    v36 = *(_QWORD *)(v33 + 32);
    v37 = (v35 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v38 = (__int64 *)(v36 + 16LL * v37);
    v39 = *v38;
    if ( v45 == *v38 )
    {
LABEL_26:
      if ( v38 != (__int64 *)(v36 + 16 * v35) )
      {
        v34 = (__int64 *)v38[1];
        goto LABEL_28;
      }
    }
    else
    {
      v42 = 1;
      while ( v39 != -8 )
      {
        v43 = v42 + 1;
        v37 = (v35 - 1) & (v42 + v37);
        v38 = (__int64 *)(v36 + 16LL * v37);
        v39 = *v38;
        if ( v45 == *v38 )
          goto LABEL_26;
        v42 = v43;
      }
    }
    v34 = 0;
LABEL_28:
    v49 = v34;
    if ( (unsigned __int8)sub_19E6B80(v47, (__int64 *)&v49, v50) )
    {
      v21 = *(char **)(a1 + 48);
      v22 = *(_DWORD *)(v50[0] + 8) <= v32;
    }
    else
    {
      v21 = *(char **)(a1 + 48);
      v22 = v9;
    }
    goto LABEL_13;
  }
  return result;
}
