// Function: sub_DBF900
// Address: 0xdbf900
//
__int64 __fastcall sub_DBF900(__int64 a1, __int16 a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  int v5; // eax
  unsigned int v8; // r15d
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v12; // rax
  int v13; // r10d
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rbx
  char v17; // bl
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // r15
  __int64 v22; // rax
  __int64 v23; // r15
  int v24; // r10d
  __int64 v25; // r9
  char v26; // r15
  int v27; // eax
  __int64 v28; // [rsp+0h] [rbp-B0h]
  __int64 v29; // [rsp+0h] [rbp-B0h]
  int v30; // [rsp+8h] [rbp-A8h]
  unsigned int v31; // [rsp+10h] [rbp-A0h]
  __int64 *v32; // [rsp+10h] [rbp-A0h]
  __int64 v33; // [rsp+10h] [rbp-A0h]
  int v34; // [rsp+18h] [rbp-98h]
  int v35; // [rsp+18h] [rbp-98h]
  __int64 *v37; // [rsp+28h] [rbp-88h]
  __int64 v38[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v39[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v40[2]; // [rsp+50h] [rbp-60h] BYREF
  const void *v41; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-48h]
  __int64 v43; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-38h]

  v5 = a5 & 6;
  v34 = a5;
  if ( v5 != 4 )
  {
    if ( v5 == 6 )
      goto LABEL_4;
    goto LABEL_3;
  }
  v32 = &a3[a4];
  v18 = (8 * a4) >> 5;
  v19 = (8 * a4) >> 3;
  if ( v18 <= 0 )
  {
    v21 = a3;
LABEL_44:
    if ( v19 != 2 )
    {
      if ( v19 != 3 )
      {
        if ( v19 != 1 )
          goto LABEL_41;
        goto LABEL_47;
      }
      if ( !(unsigned __int8)sub_DBED40(a1, *v21) )
        goto LABEL_40;
      ++v21;
    }
    if ( !(unsigned __int8)sub_DBED40(a1, *v21) )
      goto LABEL_40;
    ++v21;
LABEL_47:
    if ( !(unsigned __int8)sub_DBED40(a1, *v21) )
      goto LABEL_40;
LABEL_41:
    v34 = a5 | 6;
    v8 = a5 | 6;
    if ( a2 != 8 )
      goto LABEL_5;
    goto LABEL_15;
  }
  v20 = &a3[4 * v18];
  v21 = a3;
  v37 = v20;
  while ( (unsigned __int8)sub_DBED40(a1, *v21) )
  {
    if ( !(unsigned __int8)sub_DBED40(a1, v21[1]) )
    {
      ++v21;
      break;
    }
    if ( !(unsigned __int8)sub_DBED40(a1, v21[2]) )
    {
      v21 += 2;
      break;
    }
    if ( !(unsigned __int8)sub_DBED40(a1, v21[3]) )
    {
      v21 += 3;
      break;
    }
    v21 += 4;
    if ( v21 == v37 )
    {
      v19 = v32 - v21;
      goto LABEL_44;
    }
  }
LABEL_40:
  if ( v32 == v21 )
    goto LABEL_41;
LABEL_3:
  if ( (unsigned __int16)(a2 - 5) <= 1u )
  {
    v8 = a5;
    if ( a4 == 2 && !*(_WORD *)(*a3 + 24) )
    {
      v12 = *(_QWORD *)(*a3 + 32);
      v13 = 4 * (a2 == 6) + 13;
      v14 = v12 + 24;
      if ( (a5 & 4) == 0 )
      {
        v28 = v12 + 24;
        sub_9865C0((__int64)v38, v12 + 24);
        sub_AADBC0((__int64)&v41, v38);
        sub_AB28E0((__int64)v39, 4 * (a2 == 6) + 13, (__int64)&v41, 2);
        sub_969240(&v43);
        sub_969240((__int64 *)&v41);
        sub_969240(v38);
        v22 = sub_DBB9F0(a1, a3[1], 1u, 0);
        v23 = v22;
        v24 = 4 * (a2 == 6) + 13;
        v25 = v28;
        v42 = *(_DWORD *)(v22 + 8);
        if ( v42 > 0x40 )
        {
          sub_C43780((__int64)&v41, (const void **)v22);
          v25 = v28;
          v24 = 4 * (a2 == 6) + 13;
        }
        else
        {
          v41 = *(const void **)v22;
        }
        v44 = *(_DWORD *)(v23 + 24);
        if ( v44 > 0x40 )
        {
          v33 = v25;
          v35 = v24;
          sub_C43780((__int64)&v43, (const void **)(v23 + 16));
          v25 = v33;
          v24 = v35;
        }
        else
        {
          v43 = *(_QWORD *)(v23 + 16);
        }
        v29 = v25;
        v30 = v24;
        v26 = sub_AB1BB0((__int64)v39, (__int64)&v41);
        sub_969240(&v43);
        sub_969240((__int64 *)&v41);
        v27 = a5 | 4;
        if ( !v26 )
          v27 = a5;
        v8 = v27;
        sub_969240(v40);
        sub_969240(v39);
        v14 = v29;
        v13 = v30;
      }
      if ( (a5 & 2) == 0 )
      {
        v31 = v13;
        sub_9865C0((__int64)v38, v14);
        sub_AADBC0((__int64)&v41, v38);
        sub_AB28E0((__int64)v39, v31, (__int64)&v41, 1);
        sub_969240(&v43);
        sub_969240((__int64 *)&v41);
        sub_969240(v38);
        v15 = sub_DBB9F0(a1, a3[1], 0, 0);
        v16 = v15;
        v42 = *(_DWORD *)(v15 + 8);
        if ( v42 > 0x40 )
          sub_C43780((__int64)&v41, (const void **)v15);
        else
          v41 = *(const void **)v15;
        v44 = *(_DWORD *)(v16 + 24);
        if ( v44 > 0x40 )
          sub_C43780((__int64)&v43, (const void **)(v16 + 16));
        else
          v43 = *(_QWORD *)(v16 + 16);
        v17 = sub_AB1BB0((__int64)v39, (__int64)&v41);
        sub_969240(&v43);
        sub_969240((__int64 *)&v41);
        if ( v17 )
          v8 |= 2u;
        sub_969240(v40);
        sub_969240(v39);
      }
    }
    goto LABEL_5;
  }
LABEL_4:
  v8 = a5;
  if ( a2 != 8 )
  {
LABEL_5:
    if ( a2 == 6 && (v8 & 2) == 0 && a4 == 2 )
    {
      v9 = *a3;
      v10 = a3[1];
      if ( *(_WORD *)(*a3 + 24) == 7 && *(_QWORD *)(v9 + 40) == v10 )
        v8 |= 2u;
      if ( *(_WORD *)(v10 + 24) == 7 && v9 == *(_QWORD *)(v10 + 40) )
        v8 |= 2u;
    }
    return v8;
  }
LABEL_15:
  if ( (v34 & 1) != 0 && (v34 & 2) == 0 && a4 == 2 && sub_D968A0(*a3) && (unsigned __int8)sub_DBED40(a1, a3[1]) )
    return v34 | 2u;
  return v8;
}
