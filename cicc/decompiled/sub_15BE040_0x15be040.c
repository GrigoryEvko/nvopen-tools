// Function: sub_15BE040
// Address: 0x15be040
//
__int64 __fastcall sub_15BE040(
        __int64 *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        unsigned int a10,
        __int64 a11,
        unsigned int a12,
        __int64 a13,
        int a14,
        __int64 a15,
        __int64 a16,
        unsigned __int64 a17)
{
  __int64 v21; // rdi
  __int64 v22; // rcx
  unsigned int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 *v26; // rbx
  __int64 v27; // rdx
  unsigned int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // rdx
  __int64 v33; // r8
  int v34; // edx
  __int64 v35; // rax
  int v36; // r10d
  __int64 *v37; // r11
  int v38; // eax
  __int64 v39; // [rsp+0h] [rbp-B0h]
  __int64 v40; // [rsp+8h] [rbp-A8h]
  __int64 v43; // [rsp+28h] [rbp-88h] BYREF
  _QWORD v44[16]; // [rsp+30h] [rbp-80h] BYREF

  v21 = 0;
  if ( !(unsigned __int8)sub_16033B0() )
    return v21;
  v22 = *a1;
  v43 = a2;
  v23 = *(_DWORD *)(v22 + 1480);
  if ( !v23 )
  {
    ++*(_QWORD *)(v22 + 1456);
    goto LABEL_15;
  }
  v24 = *(_QWORD *)(v22 + 1464);
  v25 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( a2 != *v26 )
  {
    v36 = 1;
    v37 = 0;
    while ( v27 != -8 )
    {
      if ( v37 || v27 != -16 )
        v26 = v37;
      v25 = (v23 - 1) & (v36 + v25);
      v27 = *(_QWORD *)(v24 + 16LL * v25);
      if ( a2 == v27 )
      {
        v26 = (__int64 *)(v24 + 16LL * v25);
        goto LABEL_4;
      }
      ++v36;
      v37 = v26;
      v26 = (__int64 *)(v24 + 16LL * v25);
    }
    v38 = *(_DWORD *)(v22 + 1472);
    if ( v37 )
      v26 = v37;
    ++*(_QWORD *)(v22 + 1456);
    v34 = v38 + 1;
    if ( 4 * (v38 + 1) < 3 * v23 )
    {
      v33 = a2;
      if ( v23 - *(_DWORD *)(v22 + 1476) - v34 > v23 >> 3 )
        goto LABEL_17;
      v39 = v22;
      goto LABEL_16;
    }
LABEL_15:
    v39 = v22;
    v23 *= 2;
LABEL_16:
    v40 = v22 + 1456;
    sub_15B9870(v22 + 1456, v23);
    sub_15B2080(v40, &v43, v44);
    v22 = v39;
    v26 = (__int64 *)v44[0];
    v33 = v43;
    v34 = *(_DWORD *)(v39 + 1472) + 1;
LABEL_17:
    *(_DWORD *)(v22 + 1472) = v34;
    if ( *v26 != -8 )
      --*(_DWORD *)(v22 + 1476);
    *v26 = v33;
    v26[1] = 0;
    goto LABEL_20;
  }
LABEL_4:
  v21 = v26[1];
  if ( !v21 )
  {
LABEL_20:
    v35 = sub_15BDB40(a1, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a2, a17, 1u, 1);
    v26[1] = v35;
    return v35;
  }
  if ( (*(_BYTE *)(v21 + 28) & 4) != 0 && (a12 & 4) == 0 )
  {
    *(_DWORD *)(v21 + 24) = a6;
    *(_DWORD *)(v21 + 52) = a14;
    *(_WORD *)(v21 + 2) = a3;
    *(_DWORD *)(v21 + 28) = a12;
    *(_QWORD *)(v21 + 32) = a9;
    *(_DWORD *)(v21 + 48) = a10;
    *(_QWORD *)(v21 + 40) = a11;
    v44[7] = a2;
    v44[1] = a7;
    v44[2] = a4;
    v44[3] = a8;
    v44[4] = a13;
    v44[5] = a15;
    v44[6] = a16;
    v44[8] = a17;
    v21 = v26[1];
    v28 = *(_DWORD *)(v21 + 8);
    if ( v28 )
    {
      v29 = v28;
      v30 = 0;
      v31 = a5;
      while ( 1 )
      {
        if ( *(_QWORD *)(v21 + 8 * (v30 - v29)) != v31 )
        {
          sub_1623D00(v21, (unsigned int)v30);
          v21 = v26[1];
        }
        if ( v28 == (_DWORD)++v30 )
          break;
        v31 = v44[v30];
        v29 = *(unsigned int *)(v21 + 8);
      }
    }
  }
  return v21;
}
