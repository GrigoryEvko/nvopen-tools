// Function: sub_C4C400
// Address: 0xc4c400
//
unsigned __int64 *__fastcall sub_C4C400(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v8; // r8d
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // r10
  unsigned int v11; // esi
  unsigned __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rdi
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  unsigned int v21; // eax
  unsigned int v22; // edx
  unsigned __int64 v23; // rax
  unsigned __int64 v25; // rdi
  unsigned int v26; // eax
  unsigned __int64 v27; // r9
  unsigned int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-78h]
  unsigned __int64 v30; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-68h]
  unsigned __int64 v32; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-58h]
  unsigned __int64 v34; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v35; // [rsp+38h] [rbp-48h]
  unsigned __int64 v36; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v37; // [rsp+48h] [rbp-38h]

  v8 = *(_DWORD *)(a1 + 8);
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a1;
  if ( v8 > 0x40 )
    v10 = *(_QWORD *)(v9 + 8LL * ((v8 - 1) >> 6));
  v11 = *(_DWORD *)(a2 + 8);
  v12 = *(_QWORD *)a2;
  v13 = v11 - 1;
  v14 = 1LL << ((unsigned __int8)v11 - 1);
  v15 = v10 & (1LL << ((unsigned __int8)v8 - 1));
  if ( v15 )
  {
    if ( v11 > 0x40 )
    {
      if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & v14) != 0 )
      {
        v35 = v11;
        sub_C43780((__int64)&v34, (const void **)a2);
        v11 = v35;
        if ( v35 > 0x40 )
        {
          sub_C43D10((__int64)&v34);
          goto LABEL_10;
        }
        v12 = v34;
LABEL_7:
        v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~v12;
        if ( !v11 )
          v16 = 0;
        v34 = v16;
LABEL_10:
        sub_C46250((__int64)&v34);
        v17 = v35;
        v35 = 0;
        v37 = v17;
        v36 = v34;
        v18 = *(_DWORD *)(a1 + 8);
        v31 = v18;
        if ( v18 > 0x40 )
        {
          sub_C43780((__int64)&v30, (const void **)a1);
          v18 = v31;
          if ( v31 > 0x40 )
          {
            sub_C43D10((__int64)&v30);
            goto LABEL_15;
          }
          v19 = v30;
        }
        else
        {
          v19 = *(_QWORD *)a1;
        }
        v20 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v18) & ~v19;
        if ( !v18 )
          v20 = 0;
        v30 = v20;
LABEL_15:
        sub_C46250((__int64)&v30);
        v21 = v31;
        v31 = 0;
        v33 = v21;
        v32 = v30;
        sub_C4BFE0((__int64)&v32, (__int64)&v36, (_QWORD *)a3, (_QWORD *)a4);
        if ( v33 > 0x40 && v32 )
          j_j___libc_free_0_0(v32);
        if ( v31 > 0x40 && v30 )
          j_j___libc_free_0_0(v30);
        if ( v37 > 0x40 && v36 )
          j_j___libc_free_0_0(v36);
        if ( v35 > 0x40 && v34 )
          j_j___libc_free_0_0(v34);
        v22 = *(_DWORD *)(a4 + 8);
        if ( v22 <= 0x40 )
        {
LABEL_28:
          v23 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~*(_QWORD *)a4;
          if ( !v22 )
            v23 = 0;
          *(_QWORD *)a4 = v23;
          return (unsigned __int64 *)sub_C46250(a4);
        }
LABEL_64:
        sub_C43D10(a4);
        return (unsigned __int64 *)sub_C46250(a4);
      }
    }
    else if ( (v14 & v12) != 0 )
    {
      v35 = v11;
      goto LABEL_7;
    }
    v35 = v8;
    if ( v8 > 0x40 )
    {
      sub_C43780((__int64)&v34, (const void **)a1);
      v8 = v35;
      if ( v35 > 0x40 )
      {
        sub_C43D10((__int64)&v34);
        goto LABEL_55;
      }
      v9 = v34;
    }
    v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & ~v9;
    if ( !v8 )
      v27 = 0;
    v34 = v27;
LABEL_55:
    sub_C46250((__int64)&v34);
    v28 = v35;
    v35 = 0;
    v37 = v28;
    v36 = v34;
    sub_C4BFE0((__int64)&v36, a2, (_QWORD *)a3, (_QWORD *)a4);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
    {
      sub_C43D10(a3);
    }
    else
    {
      *(_QWORD *)a3 = ~*(_QWORD *)a3;
      sub_C43640((unsigned __int64 *)a3);
    }
    sub_C46250(a3);
    v22 = *(_DWORD *)(a4 + 8);
    if ( v22 <= 0x40 )
      goto LABEL_28;
    goto LABEL_64;
  }
  if ( v11 <= 0x40 )
  {
    if ( (v14 & v12) != 0 )
    {
      v35 = v11;
LABEL_36:
      v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~v12;
      if ( v11 )
        v15 = v25;
      v34 = v15;
      goto LABEL_39;
    }
    return sub_C4BFE0(a1, a2, (_QWORD *)a3, (_QWORD *)a4);
  }
  if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & v14) == 0 )
    return sub_C4BFE0(a1, a2, (_QWORD *)a3, (_QWORD *)a4);
  v35 = v11;
  v29 = v10 & (1LL << ((unsigned __int8)v8 - 1));
  sub_C43780((__int64)&v34, (const void **)a2);
  v11 = v35;
  v15 = v29;
  if ( v35 <= 0x40 )
  {
    v12 = v34;
    goto LABEL_36;
  }
  sub_C43D10((__int64)&v34);
LABEL_39:
  sub_C46250((__int64)&v34);
  v26 = v35;
  v35 = 0;
  v37 = v26;
  v36 = v34;
  sub_C4BFE0(a1, (__int64)&v36, (_QWORD *)a3, (_QWORD *)a4);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( *(_DWORD *)(a3 + 8) > 0x40u )
  {
    sub_C43D10(a3);
  }
  else
  {
    *(_QWORD *)a3 = ~*(_QWORD *)a3;
    sub_C43640((unsigned __int64 *)a3);
  }
  return (unsigned __int64 *)sub_C46250(a3);
}
