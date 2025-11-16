// Function: sub_2E0E770
// Address: 0x2e0e770
//
_QWORD *__fastcall sub_2E0E770(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v9; // rbx
  _QWORD *result; // rax
  unsigned __int64 v12; // r8
  __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // rdi
  unsigned int v16; // ecx
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // rbx
  unsigned int v22; // eax
  int v23; // eax
  unsigned __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rcx
  _QWORD *v27; // rax
  _QWORD *v28; // r8
  __int64 v29; // r10
  unsigned int v30; // eax
  bool v31; // r8
  __int64 v32; // [rsp+0h] [rbp-80h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+8h] [rbp-78h]
  unsigned __int64 v35; // [rsp+8h] [rbp-78h]
  unsigned __int64 v36; // [rsp+10h] [rbp-70h]
  unsigned __int64 v37; // [rsp+10h] [rbp-70h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  unsigned int v39; // [rsp+18h] [rbp-68h]
  _QWORD *v40; // [rsp+18h] [rbp-68h]
  __int64 *v41; // [rsp+28h] [rbp-58h] BYREF
  __int64 v42[10]; // [rsp+30h] [rbp-50h] BYREF

  v9 = *(_QWORD **)(a1 + 96);
  if ( !v9 )
  {
    v23 = *(_DWORD *)(a1 + 8);
    v41 = (__int64 *)a1;
    if ( v23 )
    {
      v24 = a5 & 0xFFFFFFFFFFFFFFF8LL;
      v25 = (a5 >> 1) & 3;
      if ( ((a5 >> 1) & 3) != 0 )
        v26 = v24 | (2LL * ((int)v25 - 1));
      else
        v26 = *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL | 6;
      v32 = a3;
      v35 = a5 & 0xFFFFFFFFFFFFFFF8LL;
      v42[0] = v26;
      v38 = v26;
      v42[1] = a5;
      v42[2] = 0;
      v27 = sub_2E09C80(a1, v42);
      v28 = v27;
      if ( v27 == *(_QWORD **)a1
        || (v29 = *(v27 - 2),
            v30 = *(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v29 >> 1) & 3,
            v30 <= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3)) )
      {
        sub_2E09930(a2, v32, a4, v38);
      }
      else
      {
        if ( v30 < (*(_DWORD *)(v35 + 24) | (unsigned int)v25) )
        {
          v40 = v28;
          if ( sub_2E09930(a2, v32, v29, v38) )
            return v9;
          sub_2E097D0(&v41, v40 - 3, a5);
          v28 = v40;
        }
        return (_QWORD *)*(v28 - 1);
      }
    }
    return v9;
  }
  v42[0] = a1;
  result = 0;
  if ( !v9[5] )
    return result;
  v12 = a5 & 0xFFFFFFFFFFFFFFF8LL;
  v39 = (a5 >> 1) & 3;
  if ( v39 )
    v13 = v12 | (2LL * (int)(v39 - 1));
  else
    v13 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v14 = (_QWORD *)v9[2];
  if ( v14 )
  {
    v15 = (__int64)(v9 + 1);
    v16 = *(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v13 >> 1) & 3;
    do
    {
      while ( 1 )
      {
        v17 = *(_DWORD *)((v14[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v14[4] >> 1) & 3;
        if ( v17 > v16
          || v17 >= v16
          && (*(_DWORD *)(v12 + 24) | v39) < (*(_DWORD *)((v14[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                            | (unsigned int)((__int64)v14[5] >> 1) & 3) )
        {
          break;
        }
        v14 = (_QWORD *)v14[3];
        if ( !v14 )
          goto LABEL_13;
      }
      v15 = (__int64)v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v14 );
LABEL_13:
    if ( v9 + 1 != (_QWORD *)v15
      && (*(_DWORD *)((*(_QWORD *)(v15 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24)
        | (unsigned int)(*(__int64 *)(v15 + 32) >> 1) & 3) <= v16 )
    {
      v33 = a3;
      v36 = v12;
      v18 = sub_220EF30(v15);
      v12 = v36;
      a3 = v33;
      v15 = v18;
    }
  }
  else
  {
    v15 = (__int64)(v9 + 1);
  }
  if ( v9[3] == v15
    || (v34 = a3,
        v37 = v12,
        v19 = sub_220EFE0(v15),
        a3 = v34,
        v21 = v19,
        v20 = *(_QWORD *)(v19 + 40),
        v22 = *(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v20 >> 1) & 3,
        v22 <= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3)) )
  {
    sub_2E09930(a2, a3, a4, v13);
    return 0;
  }
  if ( v22 >= (*(_DWORD *)(v37 + 24) | v39) )
    return *(_QWORD **)(v21 + 48);
  v31 = sub_2E09930(a2, v34, v20, v13);
  result = 0;
  if ( !v31 )
  {
    sub_2E0E620((__int64)v42, v21, a5);
    return *(_QWORD **)(v21 + 48);
  }
  return result;
}
