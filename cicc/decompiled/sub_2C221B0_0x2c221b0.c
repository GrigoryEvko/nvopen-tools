// Function: sub_2C221B0
// Address: 0x2c221b0
//
unsigned __int64 __fastcall sub_2C221B0(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rdi
  __int64 v8; // r10
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rbx
  int v17; // eax
  __int64 v18; // rsi
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rdx
  unsigned __int64 result; // rax
  int v24; // r9d
  __int64 *v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // r9
  int v32; // r11d
  unsigned int v33; // edx
  _QWORD *v34; // rcx
  __int64 v35; // r10
  _QWORD *v36; // rsi
  _QWORD *v37; // r14
  __int64 v38; // rax
  unsigned int v39; // ecx
  __int64 v40; // r8
  __int64 v41; // rsi
  unsigned int v42; // eax
  __int64 *v43; // rdx
  __int64 v44; // rdi
  int v45; // r9d
  int v46; // edx
  int v47; // r10d
  __int64 v48; // r14
  unsigned int v49; // r8d
  int v50; // esi
  int v51; // ecx
  int v52; // esi
  __int64 v53; // [rsp+8h] [rbp-78h]
  __int64 v54; // [rsp+8h] [rbp-78h]
  __int64 v55; // [rsp+10h] [rbp-70h]
  __int64 v56; // [rsp+18h] [rbp-68h]
  __int64 v57; // [rsp+18h] [rbp-68h]
  int v58; // [rsp+18h] [rbp-68h]
  __int64 v59[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v60; // [rsp+40h] [rbp-40h]

  v59[0] = *(_QWORD *)(a1 + 88);
  if ( v59[0] )
    sub_2AAAFA0(v59);
  sub_2BF1A90(a2, (__int64)v59);
  sub_9C6650(v59);
  v4 = sub_2BFB120(a2, **(_QWORD **)(a1 + 48), (unsigned int *)(a2 + 16));
  v55 = *(_QWORD *)(v4 + 40);
  v5 = sub_AA54C0(v55);
  v6 = *(_DWORD *)(a2 + 56);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = v5;
  v9 = **(_QWORD **)(a1 + 48);
  if ( v6 )
  {
    v10 = v6 - 1;
    v11 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v12 = *(_QWORD *)(v7 + 16LL * v11);
    if ( v9 == v12 )
    {
LABEL_5:
      v53 = v8;
      v56 = a2 + 32;
      v13 = sub_2BFB640(a2, v9, 0);
      v14 = *(__int64 **)(a2 + 904);
      v15 = v13;
      v60 = 257;
      v16 = sub_D5C860(v14, *(_QWORD *)(v13 + 8), 2, (__int64)v59);
      sub_F0A850(v16, *(_QWORD *)(v15 - 96), v53);
      sub_F0A850(v16, v15, v55);
      v17 = *(_DWORD *)(a2 + 56);
      v18 = a1 + 96;
      if ( v17 )
      {
        v19 = v17 - 1;
        v20 = *(_QWORD *)(a2 + 40);
        v21 = (v17 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v22 = *(_QWORD *)(v20 + 16LL * v21);
        if ( v18 == v22 )
        {
LABEL_7:
          v59[0] = a1 + 96;
          *sub_2C1C230(v56, v59) = v16;
LABEL_8:
          v59[0] = **(_QWORD **)(a1 + 48);
          result = (unsigned __int64)sub_2C1C230(v56, v59);
          *(_QWORD *)result = v16;
          return result;
        }
        v45 = 1;
        while ( v22 != -4096 )
        {
          v21 = v19 & (v45 + v21);
          v22 = *(_QWORD *)(v20 + 16LL * v21);
          if ( v18 == v22 )
            goto LABEL_7;
          ++v45;
        }
      }
      sub_2BF26E0(a2, v18, v16, 0);
      goto LABEL_8;
    }
    v24 = 1;
    while ( v12 != -4096 )
    {
      v11 = v10 & (v24 + v11);
      v12 = *(_QWORD *)(v7 + 16LL * v11);
      if ( v9 == v12 )
        goto LABEL_5;
      ++v24;
    }
  }
  v54 = v8;
  v57 = a1 + 96;
  result = sub_2C46C30(a1 + 96);
  if ( !(_BYTE)result || !*(_DWORD *)(a2 + 16) && !*(_BYTE *)(a2 + 20) )
  {
    v25 = *(__int64 **)(a2 + 904);
    v26 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a1 + 48) + 40LL) + 8LL);
    v60 = 257;
    v27 = sub_D5C860(v25, v26, 2, (__int64)v59);
    v28 = sub_ACADE0(*(__int64 ***)(v4 + 8));
    sub_F0A850(v27, v28, v54);
    sub_F0A850(v27, v4, v55);
    v29 = *(unsigned int *)(a2 + 88);
    v30 = *(_QWORD *)(a2 + 72);
    v31 = a1 + 96;
    if ( (_DWORD)v29 )
    {
      v32 = v29 - 1;
      v33 = (v29 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v34 = (_QWORD *)(v30 + 56LL * v33);
      v35 = *v34;
      v36 = v34;
      if ( v57 == *v34 )
      {
LABEL_17:
        v37 = (_QWORD *)(v30 + 56 * v29);
        if ( v37 != v36 )
        {
          v38 = *(unsigned int *)(a2 + 16);
          if ( *(_BYTE *)(a2 + 20) == 1 )
            v38 = (unsigned int)(*(_DWORD *)(a2 + 8) + v38);
          if ( (unsigned int)v38 < *((_DWORD *)v36 + 4) && *(_QWORD *)(v36[1] + 8 * v38) )
          {
            if ( v31 != v35 )
            {
              v51 = 1;
              while ( v35 != -4096 )
              {
                v52 = v51 + 1;
                v33 = v32 & (v51 + v33);
                v34 = (_QWORD *)(v30 + 56LL * v33);
                v35 = *v34;
                if ( v31 == *v34 )
                  goto LABEL_23;
                v51 = v52;
              }
              v34 = v37;
            }
LABEL_23:
            *(_QWORD *)(v34[1] + 8 * v38) = v27;
LABEL_24:
            v39 = *(_DWORD *)(a2 + 88);
            v40 = *(_QWORD *)(a2 + 72);
            v41 = **(_QWORD **)(a1 + 48);
            if ( v39 )
            {
              v42 = (v39 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v43 = (__int64 *)(v40 + 56LL * v42);
              v44 = *v43;
              if ( v41 == *v43 )
              {
LABEL_26:
                result = *(unsigned int *)(a2 + 16);
                if ( *(_BYTE *)(a2 + 20) == 1 )
                  result = (unsigned int)(*(_DWORD *)(a2 + 8) + result);
                *(_QWORD *)(v43[1] + 8 * result) = v27;
                return result;
              }
              v46 = 1;
              while ( v44 != -4096 )
              {
                v47 = v46 + 1;
                v42 = (v39 - 1) & (v46 + v42);
                v43 = (__int64 *)(v40 + 56LL * v42);
                v44 = *v43;
                if ( v41 == *v43 )
                  goto LABEL_26;
                v46 = v47;
              }
            }
            v43 = (__int64 *)(v40 + 56LL * v39);
            goto LABEL_26;
          }
        }
      }
      else
      {
        v48 = *v34;
        v49 = (v29 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v50 = 1;
        while ( v48 != -4096 )
        {
          v49 = v32 & (v50 + v49);
          v58 = v50 + 1;
          v36 = (_QWORD *)(v30 + 56LL * v49);
          v48 = *v36;
          if ( v31 == *v36 )
            goto LABEL_17;
          v50 = v58;
        }
      }
    }
    sub_2AC6E90(a2, v31, v27, (unsigned int *)(a2 + 16));
    goto LABEL_24;
  }
  return result;
}
