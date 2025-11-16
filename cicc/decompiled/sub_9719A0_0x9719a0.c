// Function: sub_9719A0
// Address: 0x9719a0
//
__int64 __fastcall sub_9719A0(unsigned int a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r12
  __int16 v11; // ax
  __int16 v12; // ax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  _BOOL4 v22; // r9d
  unsigned int v23; // eax
  unsigned __int8 v24; // bl
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // [rsp-10h] [rbp-80h]
  unsigned int v28; // [rsp+0h] [rbp-70h]
  unsigned int v29; // [rsp+4h] [rbp-6Ch]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  _BOOL4 v33; // [rsp+18h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-38h]

  v9 = (__int64)a2;
  if ( *a2 != 5 )
    goto LABEL_13;
  while ( 1 )
  {
    if ( (unsigned __int8)sub_AC30F0(a3) )
    {
      v11 = *(_WORD *)(v9 + 2);
      if ( v11 == 48 )
      {
        v17 = sub_AE4450(a4, *(_QWORD *)(v9 + 8));
        v18 = sub_96F3F0(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)), v17, 0, a4);
        if ( v18 )
        {
          v31 = v18;
          a3 = sub_AD6530(*(_QWORD *)(v18 + 8));
          v9 = v31;
          goto LABEL_12;
        }
        v11 = *(_WORD *)(v9 + 2);
      }
      if ( v11 == 47
        && *(_QWORD *)(v9 + 8) == sub_AE4450(
                                    a4,
                                    *(_QWORD *)(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)) + 8LL)) )
      {
        v9 = *(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
        a3 = sub_AD6530(*(_QWORD *)(v9 + 8));
        goto LABEL_12;
      }
    }
    if ( *(_BYTE *)a3 != 5 )
      goto LABEL_15;
    v12 = *(_WORD *)(v9 + 2);
    if ( *(_WORD *)(a3 + 2) != v12 )
      goto LABEL_15;
    if ( v12 != 48 )
      goto LABEL_8;
    v19 = sub_AE4450(a4, *(_QWORD *)(v9 + 8));
    v32 = sub_96F3F0(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)), v19, 0, a4);
    v20 = sub_96F3F0(*(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)), v19, 0, a4);
    if ( !v32 || !v20 )
      break;
    a3 = v20;
    v9 = v32;
LABEL_12:
    while ( 1 )
    {
      a6 = 0;
      if ( *(_BYTE *)v9 == 5 )
        break;
LABEL_13:
      if ( *(_BYTE *)a3 != 5 )
        goto LABEL_17;
      a1 = sub_B52F50(a1);
      v15 = v9;
      v9 = a3;
      a3 = v15;
    }
  }
  v12 = *(_WORD *)(v9 + 2);
LABEL_8:
  if ( v12 != 47 )
  {
LABEL_15:
    v14 = *(_QWORD *)(v9 + 8);
    goto LABEL_16;
  }
  v13 = sub_AE4450(a4, *(_QWORD *)(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)) + 8LL));
  v14 = *(_QWORD *)(v9 + 8);
  if ( v14 == v13
    && *(_QWORD *)(*(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)) + 8LL) == *(_QWORD *)(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF))
                                                                                                  + 8LL) )
  {
    a3 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    v9 = *(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
    goto LABEL_12;
  }
LABEL_16:
  if ( *(_BYTE *)(v14 + 8) != 14 || (unsigned __int8)sub_B532B0(a1) )
    goto LABEL_17;
  v37 = sub_AE43F0(a4, *(_QWORD *)(v9 + 8));
  if ( v37 > 0x40 )
  {
    v28 = v37;
    sub_C43690(&v36, 0, 0);
    v30 = sub_BD45C0(v9, a4, (unsigned int)&v36, a1 - 32 <= 1, 0, a1 - 32 <= 1, 0, 0);
    v39 = v28;
    sub_C43690(&v38, 0, 0);
    v22 = a1 - 32 <= 1;
  }
  else
  {
    v29 = v37;
    v33 = a1 - 32 <= 1;
    v36 = 0;
    v21 = sub_BD45C0(v9, a4, (unsigned int)&v36, v33, 0, v33, 0, 0);
    v38 = 0;
    v30 = v21;
    v22 = v33;
    v39 = v29;
  }
  if ( v30 != sub_BD45C0(a3, a4, (unsigned int)&v38, v22, 0, v22, 0, 0) )
  {
    if ( v39 > 0x40 && v38 )
      j_j___libc_free_0_0(v38);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
LABEL_17:
    if ( a1 > 0xF )
      return sub_AAB310(a1, v9, a3);
    v9 = sub_96ED60(v9, a6, 0);
    if ( v9 && (a3 = sub_96ED60(a3, a6, 0)) != 0 )
      return sub_AAB310(a1, v9, a3);
    else
      return 0;
  }
  v23 = sub_B52E90(a1, a4, v27);
  v24 = sub_B532C0(&v36, &v38, v23);
  v26 = sub_BD5C60(v9, &v38, v25);
  result = sub_ACD760(v26, v24);
  if ( v39 > 0x40 && v38 )
  {
    v34 = result;
    j_j___libc_free_0_0(v38);
    result = v34;
  }
  if ( v37 > 0x40 && v36 )
  {
    v35 = result;
    j_j___libc_free_0_0(v36);
    return v35;
  }
  return result;
}
