// Function: sub_2A028D0
// Address: 0x2a028d0
//
_QWORD *__fastcall sub_2A028D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v5; // r9
  __int64 v7; // r14
  _QWORD *v10; // rdi
  __int64 v11; // rax
  _QWORD *v12; // r13
  _BYTE *v13; // rsi
  __int64 v14; // r9
  __int64 *v15; // r11
  __int64 *v16; // rbx
  __int64 v17; // rcx
  __int64 *v18; // r13
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // rdi
  int v23; // esi
  __int64 v24; // r8
  int v25; // esi
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r10
  unsigned int v29; // esi
  int v30; // edx
  __int64 v31; // rax
  _QWORD *v32; // r8
  int v33; // edi
  __int64 v34; // rdx
  unsigned __int64 *v35; // rdi
  __int64 v36; // rdx
  _QWORD *v37; // rdx
  __int64 *v38; // rbx
  __int64 *i; // r12
  __int64 v40; // rsi
  __int64 v42; // r10
  unsigned int v43; // r9d
  _QWORD *v44; // rdx
  __int64 v45; // rdi
  int v46; // eax
  int v47; // r9d
  __int64 v48; // rax
  _BYTE *v49; // rsi
  __int64 v50; // rax
  int v51; // edi
  int v52; // esi
  int v53; // esi
  __int64 v54; // r10
  _QWORD *v55; // r9
  int v56; // r11d
  unsigned int v57; // edx
  __int64 v58; // rdi
  int v59; // edx
  __int64 v60; // r10
  int v61; // r11d
  unsigned int v62; // esi
  __int64 v63; // rdi
  __int64 v64; // [rsp+8h] [rbp-88h]
  _QWORD *v65; // [rsp+10h] [rbp-80h]
  __int64 v66; // [rsp+10h] [rbp-80h]
  __int64 v67; // [rsp+10h] [rbp-80h]
  __int64 v68; // [rsp+20h] [rbp-70h]
  __int64 v69; // [rsp+20h] [rbp-70h]
  unsigned __int64 *v70; // [rsp+20h] [rbp-70h]
  _QWORD *v71; // [rsp+20h] [rbp-70h]
  _QWORD *v72; // [rsp+20h] [rbp-70h]
  __int64 v73; // [rsp+20h] [rbp-70h]
  int v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+28h] [rbp-68h]
  __int64 *v77; // [rsp+28h] [rbp-68h]
  __int64 v78; // [rsp+28h] [rbp-68h]
  __int64 v79; // [rsp+28h] [rbp-68h]
  void *v80; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v81[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v82; // [rsp+48h] [rbp-48h]
  __int64 v83; // [rsp+50h] [rbp-40h]

  v5 = a2;
  v7 = a1;
  v10 = *(_QWORD **)(a1 + 32);
  v10[17] += 160LL;
  v11 = v10[7];
  v12 = (_QWORD *)((v11 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v10[8] >= (unsigned __int64)(v12 + 20) && v11 )
  {
    v10[7] = v12 + 20;
  }
  else
  {
    v50 = sub_9D1E70((__int64)(v10 + 7), 160, 160, 3);
    v5 = a2;
    v12 = (_QWORD *)v50;
  }
  memset(v12, 0, 0xA0u);
  v12[9] = 8;
  v12[8] = v12 + 11;
  *((_BYTE *)v12 + 84) = 1;
  if ( a3 )
  {
    v80 = v12;
    *v12 = a3;
    v13 = *(_BYTE **)(a3 + 16);
    if ( v13 == *(_BYTE **)(a3 + 24) )
    {
      v78 = v5;
      sub_D4C7F0(a3 + 8, v13, &v80);
      v5 = v78;
    }
    else
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = v80;
        v13 = *(_BYTE **)(a3 + 16);
      }
      *(_QWORD *)(a3 + 16) = v13 + 8;
    }
  }
  else
  {
    v48 = *(_QWORD *)(v7 + 32);
    v80 = v12;
    v49 = *(_BYTE **)(v48 + 40);
    if ( v49 == *(_BYTE **)(v48 + 48) )
    {
      v79 = v5;
      sub_D4C7F0(v48 + 32, v49, &v80);
      v5 = v79;
    }
    else
    {
      if ( v49 )
      {
        *(_QWORD *)v49 = v12;
        v49 = *(_BYTE **)(v48 + 40);
      }
      *(_QWORD *)(v48 + 40) = v49 + 8;
    }
  }
  v76 = v5;
  (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(v7 + 40))(*(_QWORD *)(v7 + 48), v12, a5);
  v14 = v76;
  v15 = *(__int64 **)(v76 + 40);
  v16 = *(__int64 **)(v76 + 32);
  if ( v16 != v15 )
  {
    v77 = v12;
    v17 = a4;
    v18 = v15;
    v19 = v7;
    v20 = v14;
    while ( 1 )
    {
      v21 = *(_QWORD *)(v19 + 32);
      v22 = *v16;
      v23 = *(_DWORD *)(v21 + 24);
      v24 = *(_QWORD *)(v21 + 8);
      if ( v23 )
      {
        v25 = v23 - 1;
        v26 = v25 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        if ( v22 != *v27 )
        {
          v46 = 1;
          while ( v28 != -4096 )
          {
            v47 = v46 + 1;
            v26 = v25 & (v46 + v26);
            v27 = (__int64 *)(v24 + 16LL * v26);
            v28 = *v27;
            if ( v22 == *v27 )
              goto LABEL_14;
            v46 = v47;
          }
          goto LABEL_11;
        }
LABEL_14:
        if ( v20 == v27[1] )
          break;
      }
LABEL_11:
      if ( v18 == ++v16 )
      {
        v12 = v77;
        v14 = v20;
        v7 = v19;
        a4 = v17;
        goto LABEL_33;
      }
    }
    v81[0] = 2;
    v81[1] = 0;
    v82 = v22;
    if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
    {
      v68 = v17;
      sub_BD73F0((__int64)v81);
      v17 = v68;
    }
    v29 = *(_DWORD *)(v17 + 24);
    v83 = v17;
    v80 = &unk_49DD7B0;
    if ( v29 )
    {
      v31 = v82;
      v42 = *(_QWORD *)(v17 + 8);
      v43 = (v29 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
      v44 = (_QWORD *)(v42 + ((unsigned __int64)v43 << 6));
      v45 = v44[3];
      if ( v82 == v45 )
      {
LABEL_37:
        v37 = v44 + 5;
LABEL_38:
        v80 = &unk_49DB368;
        if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        {
          v67 = v17;
          v72 = v37;
          sub_BD60C0(v81);
          v17 = v67;
          v37 = v72;
        }
        v73 = v17;
        sub_D4F330(v77, v37[2], v21);
        v17 = v73;
        goto LABEL_11;
      }
      v74 = 1;
      v32 = 0;
      while ( v45 != -4096 )
      {
        if ( !v32 && v45 == -8192 )
          v32 = v44;
        v43 = (v29 - 1) & (v74 + v43);
        v44 = (_QWORD *)(v42 + ((unsigned __int64)v43 << 6));
        v45 = v44[3];
        if ( v82 == v45 )
          goto LABEL_37;
        ++v74;
      }
      v51 = *(_DWORD *)(v17 + 16);
      if ( !v32 )
        v32 = v44;
      ++*(_QWORD *)v17;
      v33 = v51 + 1;
      if ( 4 * v33 < 3 * v29 )
      {
        if ( v29 - *(_DWORD *)(v17 + 20) - v33 > v29 >> 3 )
        {
LABEL_23:
          *(_DWORD *)(v17 + 16) = v33;
          if ( v32[3] == -4096 )
          {
            v35 = v32 + 1;
            if ( v31 != -4096 )
            {
LABEL_28:
              v32[3] = v31;
              if ( v31 == 0 || v31 == -4096 || v31 == -8192 )
              {
                v31 = v82;
              }
              else
              {
                v66 = v17;
                v71 = v32;
                sub_BD6050(v35, v81[0] & 0xFFFFFFFFFFFFFFF8LL);
                v31 = v82;
                v17 = v66;
                v32 = v71;
              }
            }
          }
          else
          {
            --*(_DWORD *)(v17 + 20);
            v34 = v32[3];
            if ( v34 != v31 )
            {
              v35 = v32 + 1;
              if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
              {
                v64 = v17;
                v65 = v32;
                v70 = v32 + 1;
                sub_BD60C0(v35);
                v31 = v82;
                v17 = v64;
                v32 = v65;
                v35 = v70;
              }
              goto LABEL_28;
            }
          }
          v36 = v83;
          v32[5] = 6;
          v32[6] = 0;
          v32[4] = v36;
          v37 = v32 + 5;
          v32[7] = 0;
          goto LABEL_38;
        }
        v75 = v17;
        sub_CF32C0(v17, v29);
        v17 = v75;
        v52 = *(_DWORD *)(v75 + 24);
        if ( !v52 )
          goto LABEL_21;
        v31 = v82;
        v53 = v52 - 1;
        v54 = *(_QWORD *)(v75 + 8);
        v55 = 0;
        v56 = 1;
        v57 = v53 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
        v32 = (_QWORD *)(v54 + ((unsigned __int64)v57 << 6));
        v58 = v32[3];
        if ( v82 == v58 )
          goto LABEL_22;
        while ( v58 != -4096 )
        {
          if ( !v55 && v58 == -8192 )
            v55 = v32;
          v57 = v53 & (v56 + v57);
          v32 = (_QWORD *)(v54 + ((unsigned __int64)v57 << 6));
          v58 = v32[3];
          if ( v82 == v58 )
            goto LABEL_22;
          ++v56;
        }
        goto LABEL_72;
      }
    }
    else
    {
      ++*(_QWORD *)v17;
    }
    v69 = v17;
    sub_CF32C0(v17, 2 * v29);
    v17 = v69;
    v30 = *(_DWORD *)(v69 + 24);
    if ( !v30 )
    {
LABEL_21:
      v31 = v82;
      v32 = 0;
LABEL_22:
      v33 = *(_DWORD *)(v17 + 16) + 1;
      goto LABEL_23;
    }
    v31 = v82;
    v59 = v30 - 1;
    v60 = *(_QWORD *)(v69 + 8);
    v55 = 0;
    v61 = 1;
    v62 = v59 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
    v32 = (_QWORD *)(v60 + ((unsigned __int64)v62 << 6));
    v63 = v32[3];
    if ( v82 == v63 )
      goto LABEL_22;
    while ( v63 != -4096 )
    {
      if ( !v55 && v63 == -8192 )
        v55 = v32;
      v62 = v59 & (v61 + v62);
      v32 = (_QWORD *)(v60 + ((unsigned __int64)v62 << 6));
      v63 = v32[3];
      if ( v82 == v63 )
        goto LABEL_22;
      ++v61;
    }
LABEL_72:
    if ( v55 )
      v32 = v55;
    goto LABEL_22;
  }
LABEL_33:
  v38 = *(__int64 **)(v14 + 16);
  for ( i = *(__int64 **)(v14 + 8); v38 != i; ++i )
  {
    v40 = *i;
    sub_2A028D0(v7, v40, v12, a4, 1);
  }
  return v12;
}
