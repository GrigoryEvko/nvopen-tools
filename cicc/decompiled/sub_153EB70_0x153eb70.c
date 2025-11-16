// Function: sub_153EB70
// Address: 0x153eb70
//
__int64 __fastcall sub_153EB70(__int64 a1, __int64 a2)
{
  int v3; // edx
  int *v4; // rax
  unsigned int v5; // ecx
  int v6; // edi
  __int64 v7; // r8
  int v8; // esi
  __int64 v9; // rdx
  __int64 v10; // r12
  int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  _BYTE *v15; // r14
  const void *v16; // r8
  __int64 result; // rax
  _BYTE *v18; // rcx
  signed __int64 v19; // r12
  _BYTE *v20; // r9
  unsigned __int64 v21; // rax
  size_t v22; // rdx
  unsigned __int64 v23; // r13
  bool v24; // cf
  unsigned __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rax
  char *v28; // r13
  char *v29; // r11
  char *v30; // r15
  _BYTE *v31; // r9
  _BYTE *v32; // rcx
  size_t v33; // rax
  size_t v34; // r12
  char *v35; // r15
  size_t v36; // rdx
  unsigned int v37; // r9d
  char *dest; // [rsp+8h] [rbp-58h]
  _BYTE *v39; // [rsp+10h] [rbp-50h]
  void *src; // [rsp+18h] [rbp-48h]
  _BYTE *srca; // [rsp+18h] [rbp-48h]
  _BYTE *srcb; // [rsp+18h] [rbp-48h]
  void *v43; // [rsp+20h] [rbp-40h]
  _BYTE *v44; // [rsp+20h] [rbp-40h]
  _BYTE *v45; // [rsp+20h] [rbp-40h]
  _BYTE *v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+28h] [rbp-38h]

  *(_DWORD *)(a1 + 540) = (__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 3;
  v3 = sub_153E840(a1, a2) + 1;
  if ( (*(_BYTE *)(a1 + 296) & 1) != 0 )
  {
    v4 = (int *)(a1 + 304);
    v5 = 0;
    v6 = 0;
    v7 = a1 + 304;
  }
  else
  {
    result = *(unsigned int *)(a1 + 312);
    v7 = *(_QWORD *)(a1 + 304);
    if ( !(_DWORD)result )
    {
LABEL_26:
      *(_DWORD *)(a1 + 544) = 0;
      return result;
    }
    v6 = result - 1;
    v5 = (result - 1) & (37 * v3);
    v4 = (int *)(v7 + 16LL * v5);
  }
  v8 = *v4;
  if ( v3 != *v4 )
  {
    result = 1;
    while ( v8 != -1 )
    {
      v37 = result + 1;
      v5 = v6 & (result + v5);
      v4 = (int *)(v7 + 16LL * v5);
      v8 = *v4;
      if ( v3 == *v4 )
        goto LABEL_4;
      result = v37;
    }
    goto LABEL_26;
  }
LABEL_4:
  v9 = (unsigned int)v4[1];
  v10 = (unsigned int)v4[2];
  v11 = v4[3];
  v12 = *(_QWORD *)(a1 + 232);
  v13 = 8 * v9;
  v14 = 8 * v10;
  v15 = *(_BYTE **)(a1 + 216);
  v16 = (const void *)(v12 + v13);
  result = v14 + v12;
  *(_DWORD *)(a1 + 544) = v11;
  if ( v16 == (const void *)result )
    return result;
  v18 = *(_BYTE **)(a1 + 224);
  v19 = v14 - v13;
  if ( v18 - v15 < (unsigned __int64)v19 )
  {
    v20 = *(_BYTE **)(a1 + 208);
    v21 = v19 >> 3;
    v22 = v15 - v20;
    v23 = (v15 - v20) >> 3;
    if ( v19 >> 3 > 0xFFFFFFFFFFFFFFFLL - v23 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v21 < v23 )
      v21 = (v15 - v20) >> 3;
    v24 = __CFADD__(v23, v21);
    v25 = v23 + v21;
    if ( v24 )
    {
      v26 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v25 )
      {
        v47 = 0;
        v28 = 0;
LABEL_19:
        v29 = &v28[v22];
        v30 = &v28[v22 + v19];
        if ( v15 == v20 )
        {
          v36 = v19;
          srcb = v18;
          v46 = v20;
          v34 = 0;
          memcpy(v29, v16, v36);
          v31 = v46;
          v32 = srcb;
          v33 = *(_QWORD *)(a1 + 216) - (_QWORD)v15;
          if ( *(_BYTE **)(a1 + 216) == v15 )
          {
LABEL_22:
            v35 = &v30[v34];
            if ( !v31 )
            {
LABEL_23:
              *(_QWORD *)(a1 + 208) = v28;
              *(_QWORD *)(a1 + 216) = v35;
              *(_QWORD *)(a1 + 224) = v47;
              return v47;
            }
LABEL_29:
            j_j___libc_free_0(v31, v32 - v31);
            goto LABEL_23;
          }
        }
        else
        {
          v39 = v18;
          v44 = v20;
          dest = &v28[v22];
          src = (void *)v16;
          memmove(v28, v20, v22);
          memcpy(dest, src, v19);
          v31 = v44;
          v32 = v39;
          v33 = *(_QWORD *)(a1 + 216) - (_QWORD)v15;
          if ( v15 == *(_BYTE **)(a1 + 216) )
          {
            v35 = &v30[v33];
            goto LABEL_29;
          }
        }
        srca = v32;
        v45 = v31;
        v34 = v33;
        memcpy(v30, v15, v33);
        v32 = srca;
        v31 = v45;
        goto LABEL_22;
      }
      if ( v25 > 0xFFFFFFFFFFFFFFFLL )
        v25 = 0xFFFFFFFFFFFFFFFLL;
      v26 = 8 * v25;
    }
    v43 = (void *)v16;
    v27 = sub_22077B0(v26);
    v20 = *(_BYTE **)(a1 + 208);
    v16 = v43;
    v28 = (char *)v27;
    v18 = *(_BYTE **)(a1 + 224);
    v47 = v26 + v27;
    v22 = v15 - v20;
    goto LABEL_19;
  }
  result = (__int64)memmove(v15, v16, v19);
  *(_QWORD *)(a1 + 216) += v19;
  return result;
}
