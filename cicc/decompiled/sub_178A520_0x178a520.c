// Function: sub_178A520
// Address: 0x178a520
//
__int64 **__fastcall sub_178A520(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  int v3; // r9d
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r8
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r8
  unsigned __int64 *v10; // rax
  __int64 **v11; // r15
  unsigned __int64 *v13; // r14
  __int64 *v14; // r12
  unsigned __int8 v15; // al
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r13
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // r15
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rbx
  int v30; // eax
  __int64 v31; // rax
  int v32; // edx
  __int64 v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // rdi
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 ***v40; // r13
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // [rsp+0h] [rbp-D0h]
  __int64 v46; // [rsp+8h] [rbp-C8h]
  int v47; // [rsp+18h] [rbp-B8h]
  unsigned int v48; // [rsp+20h] [rbp-B0h]
  __int64 v49; // [rsp+20h] [rbp-B0h]
  int v51; // [rsp+38h] [rbp-98h]
  __int64 v52; // [rsp+38h] [rbp-98h]
  _QWORD v53[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v54[2]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v55; // [rsp+60h] [rbp-70h]
  _BYTE *v56; // [rsp+70h] [rbp-60h] BYREF
  __int64 v57; // [rsp+78h] [rbp-58h]
  _BYTE v58[80]; // [rsp+80h] [rbp-50h] BYREF

  v2 = sub_157EBA0(*(_QWORD *)(a2 + 40));
  if ( v2 )
  {
    v4 = (unsigned int)*(unsigned __int8 *)(v2 + 16) - 34;
    if ( (unsigned int)v4 <= 0x36 )
    {
      v5 = 0x40018000000001LL;
      if ( _bittest64(&v5, v4) )
        return 0;
    }
  }
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v47 = v6;
  if ( (unsigned int)v6 <= 2 )
    return 0;
  v46 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = 3 * v6;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v8 = *(unsigned __int64 **)(a2 - 8);
    v9 = &v8[v7];
  }
  else
  {
    v8 = (unsigned __int64 *)(a2 - v7 * 8);
    v9 = (unsigned __int64 *)a2;
  }
  v10 = v8;
  while ( *(_BYTE *)(*v10 + 16) != 61 )
  {
    v10 += 3;
    if ( v9 == v10 )
      return 0;
  }
  v11 = **(__int64 ****)(*v10 - 24);
  if ( v11 )
  {
    v45 = a2;
    v13 = v9;
    v56 = v58;
    v57 = 0x400000000LL;
    v51 = 0;
    v48 = 0;
    do
    {
      v14 = (__int64 *)*v8;
      v15 = *(_BYTE *)(*v8 + 16);
      if ( v15 <= 0x17u )
      {
        if ( v15 > 0x10u )
          goto LABEL_17;
        v16 = sub_15A43B0(*v8, v11, 0);
        a2 = *v14;
        v17 = v16;
        if ( v14 != (__int64 *)sub_15A3CB0(v16, (__int64 **)*v14, 0) )
          goto LABEL_17;
        v18 = (unsigned int)v57;
        if ( (unsigned int)v57 >= HIDWORD(v57) )
        {
          a2 = (__int64)v58;
          sub_16CD150((__int64)&v56, v58, 0, 8, (int)v9, v3);
          v18 = (unsigned int)v57;
        }
        ++v51;
        *(_QWORD *)&v56[8 * v18] = v17;
        LODWORD(v57) = v57 + 1;
      }
      else
      {
        if ( v15 != 61 )
          goto LABEL_17;
        v40 = (__int64 ***)*(v14 - 3);
        if ( v11 != *v40 )
          goto LABEL_17;
        v41 = v14[1];
        if ( !v41 || *(_QWORD *)(v41 + 8) )
          goto LABEL_17;
        v42 = (unsigned int)v57;
        if ( (unsigned int)v57 >= HIDWORD(v57) )
        {
          a2 = (__int64)v58;
          sub_16CD150((__int64)&v56, v58, 0, 8, (int)v9, v3);
          v42 = (unsigned int)v57;
        }
        ++v48;
        *(_QWORD *)&v56[8 * v42] = v40;
        LODWORD(v57) = v57 + 1;
      }
      v8 += 3;
    }
    while ( v13 != v8 );
    if ( !v51 || v48 <= 1 )
    {
LABEL_17:
      v11 = 0;
      goto LABEL_18;
    }
    v53[0] = sub_1649960(v45);
    v53[1] = v19;
    v54[0] = (__int64)v53;
    v55 = 773;
    v54[1] = (__int64)".shrunk";
    v20 = sub_1648B60(64);
    v22 = v20;
    if ( v20 )
    {
      v23 = v20;
      sub_15F1EA0(v20, (__int64)v11, 53, 0, 0, 0);
      *(_DWORD *)(v22 + 56) = v47;
      sub_164B780(v22, v54);
      a2 = *(unsigned int *)(v22 + 56);
      sub_1648880(v22, a2, 1);
    }
    else
    {
      v23 = 0;
    }
    v24 = 0;
    v25 = v23;
    v26 = 8 * v46;
    do
    {
      if ( (*(_BYTE *)(v45 + 23) & 0x40) != 0 )
        v27 = *(_QWORD *)(v45 - 8);
      else
        v27 = v45 - 24LL * (*(_DWORD *)(v45 + 20) & 0xFFFFFFF);
      v28 = *(_QWORD *)(v24 + v27 + 24LL * *(unsigned int *)(v45 + 56) + 8);
      v29 = *(_QWORD *)&v56[v24];
      v30 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
      if ( v30 == *(_DWORD *)(v22 + 56) )
      {
        v49 = v26;
        v52 = *(_QWORD *)(v24 + v27 + 24LL * *(unsigned int *)(v45 + 56) + 8);
        sub_15F55D0(v22, a2, v27, v28, v21, v26);
        v26 = v49;
        v28 = v52;
        v30 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
      }
      v31 = (v30 + 1) & 0xFFFFFFF;
      v32 = v31 | *(_DWORD *)(v22 + 20) & 0xF0000000;
      *(_DWORD *)(v22 + 20) = v32;
      if ( (v32 & 0x40000000) != 0 )
        v33 = *(_QWORD *)(v22 - 8);
      else
        v33 = v25 - 24 * v31;
      v34 = (_QWORD *)(v33 + 24LL * (unsigned int)(v31 - 1));
      if ( *v34 )
      {
        v35 = v34[1];
        v36 = v34[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v36 = v35;
        if ( v35 )
        {
          v21 = *(_QWORD *)(v35 + 16) & 3LL;
          *(_QWORD *)(v35 + 16) = v21 | v36;
        }
      }
      *v34 = v29;
      if ( v29 )
      {
        v37 = *(_QWORD *)(v29 + 8);
        v21 = v29 + 8;
        v34[1] = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = (unsigned __int64)(v34 + 1) | *(_QWORD *)(v37 + 16) & 3LL;
        v34[2] = v21 | v34[2] & 3LL;
        *(_QWORD *)(v29 + 8) = v34;
      }
      v38 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
        v39 = *(_QWORD *)(v22 - 8);
      else
        v39 = v25 - 24 * v38;
      v24 += 8;
      *(_QWORD *)(v39 + 8LL * (unsigned int)(v38 - 1) + 24LL * *(unsigned int *)(v22 + 56) + 8) = v28;
    }
    while ( v26 != v24 );
    sub_157E9D0(*(_QWORD *)(v45 + 40) + 40LL, v22);
    v43 = *(_QWORD *)(v45 + 24);
    *(_QWORD *)(v22 + 32) = v45 + 24;
    v43 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v22 + 24) = v43 | *(_QWORD *)(v22 + 24) & 7LL;
    *(_QWORD *)(v43 + 8) = v22 + 24;
    *(_QWORD *)(v45 + 24) = *(_QWORD *)(v45 + 24) & 7LL | (v22 + 24);
    sub_170B990(*a1, v22);
    v44 = *(_QWORD *)v45;
    v55 = 257;
    v11 = (__int64 **)sub_15FDE70((_QWORD *)v22, v44, (__int64)v54, 0);
LABEL_18:
    if ( v56 != v58 )
      _libc_free((unsigned __int64)v56);
  }
  return v11;
}
