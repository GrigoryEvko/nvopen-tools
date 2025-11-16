// Function: sub_1478130
// Address: 0x1478130
//
__int64 __fastcall sub_1478130(__int64 a1, int a2, __int64 **a3, unsigned int a4)
{
  int v4; // eax
  unsigned int v6; // r12d
  __int64 v8; // rax
  int v9; // r9d
  __int64 v10; // r10
  __int64 *v11; // r13
  char v12; // r13
  __int64 *v13; // r15
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 *v17; // rbx
  __int64 *v18; // rax
  int v19; // r9d
  __int64 v20; // r10
  __int64 v21; // [rsp+8h] [rbp-A8h]
  __int64 v22; // [rsp+8h] [rbp-A8h]
  int v23; // [rsp+10h] [rbp-A0h]
  __int64 v24; // [rsp+10h] [rbp-A0h]
  unsigned int v25; // [rsp+18h] [rbp-98h]
  __int64 *v26; // [rsp+18h] [rbp-98h]
  char v27; // [rsp+18h] [rbp-98h]
  int v28; // [rsp+18h] [rbp-98h]
  __int64 *v29; // [rsp+18h] [rbp-98h]
  char v30; // [rsp+20h] [rbp-90h]
  __int64 v31[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v32[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v33[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v35; // [rsp+68h] [rbp-48h]
  __int64 v36; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v37; // [rsp+78h] [rbp-38h]

  v4 = a4 & 6;
  v6 = a4;
  v30 = a4;
  if ( v4 != 4 )
  {
    if ( v4 == 6 )
      return v6;
    goto LABEL_3;
  }
  v13 = *a3;
  v14 = 8LL * *((unsigned int *)a3 + 2);
  v26 = &(*a3)[(unsigned __int64)v14 / 8];
  v15 = v14 >> 3;
  v16 = v14 >> 5;
  if ( v16 )
  {
    v17 = &v13[4 * v16];
    while ( (unsigned __int8)sub_1477BC0(a1, *v13) )
    {
      if ( !(unsigned __int8)sub_1477BC0(a1, v13[1]) )
      {
        ++v13;
        goto LABEL_23;
      }
      if ( !(unsigned __int8)sub_1477BC0(a1, v13[2]) )
      {
        v13 += 2;
        goto LABEL_23;
      }
      if ( !(unsigned __int8)sub_1477BC0(a1, v13[3]) )
      {
        v13 += 3;
        goto LABEL_23;
      }
      v13 += 4;
      if ( v17 == v13 )
      {
        v15 = v26 - v13;
        goto LABEL_26;
      }
    }
    goto LABEL_23;
  }
LABEL_26:
  if ( v15 == 2 )
    goto LABEL_36;
  if ( v15 == 3 )
  {
    if ( !(unsigned __int8)sub_1477BC0(a1, *v13) )
      goto LABEL_23;
    ++v13;
LABEL_36:
    if ( !(unsigned __int8)sub_1477BC0(a1, *v13) )
      goto LABEL_23;
    ++v13;
    goto LABEL_29;
  }
  if ( v15 != 1 )
    goto LABEL_24;
LABEL_29:
  if ( (unsigned __int8)sub_1477BC0(a1, *v13) )
    goto LABEL_24;
LABEL_23:
  if ( v26 == v13 )
  {
LABEL_24:
    v6 |= 6u;
    return v6;
  }
LABEL_3:
  if ( (unsigned int)(a2 - 4) <= 1 && *((_DWORD *)a3 + 2) == 2 )
  {
    v8 = **a3;
    if ( !*(_WORD *)(v8 + 24) )
    {
      v9 = 4 * (a2 == 5) + 11;
      v10 = *(_QWORD *)(v8 + 32) + 24LL;
      if ( (v6 & 4) == 0 )
      {
        v21 = *(_QWORD *)(v8 + 32) + 24LL;
        sub_13A38D0((__int64)v31, v21);
        sub_1589870(&v34, v31);
        sub_1591060(v32, 4 * (unsigned int)(a2 == 5) + 11, &v34, 2);
        sub_135E100(&v36);
        sub_135E100(&v34);
        sub_135E100(v31);
        v18 = sub_1477920(a1, (*a3)[1], 1u);
        v19 = 4 * (a2 == 5) + 11;
        v20 = v21;
        v35 = *((_DWORD *)v18 + 2);
        if ( v35 > 0x40 )
        {
          v29 = v18;
          sub_16A4FD0(&v34, v18);
          v20 = v21;
          v19 = 4 * (a2 == 5) + 11;
          v18 = v29;
        }
        else
        {
          v34 = *v18;
        }
        v37 = *((_DWORD *)v18 + 6);
        if ( v37 > 0x40 )
        {
          v24 = v20;
          v28 = v19;
          sub_16A4FD0(&v36, v18 + 2);
          v20 = v24;
          v19 = v28;
        }
        else
        {
          v36 = v18[2];
        }
        v22 = v20;
        v23 = v19;
        v27 = sub_158BB40(v32, &v34);
        sub_135E100(&v36);
        sub_135E100(&v34);
        if ( v27 )
          v6 |= 4u;
        sub_135E100(v33);
        sub_135E100(v32);
        v10 = v22;
        v9 = v23;
      }
      if ( (v30 & 2) == 0 )
      {
        v25 = v9;
        sub_13A38D0((__int64)v31, v10);
        sub_1589870(&v34, v31);
        sub_1591060(v32, v25, &v34, 1);
        sub_135E100(&v36);
        sub_135E100(&v34);
        sub_135E100(v31);
        v11 = sub_1477920(a1, (*a3)[1], 0);
        v35 = *((_DWORD *)v11 + 2);
        if ( v35 > 0x40 )
          sub_16A4FD0(&v34, v11);
        else
          v34 = *v11;
        v37 = *((_DWORD *)v11 + 6);
        if ( v37 > 0x40 )
          sub_16A4FD0(&v36, v11 + 2);
        else
          v36 = v11[2];
        v12 = sub_158BB40(v32, &v34);
        sub_135E100(&v36);
        sub_135E100(&v34);
        if ( v12 )
          v6 |= 2u;
        sub_135E100(v33);
        sub_135E100(v32);
      }
    }
  }
  return v6;
}
