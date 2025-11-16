// Function: sub_1699DC0
// Address: 0x1699dc0
//
__int64 __fastcall sub_1699DC0(
        __int64 a1,
        void *a2,
        __int64 a3,
        unsigned int a4,
        unsigned __int8 a5,
        unsigned int a6,
        _BYTE *a7)
{
  char v7; // al
  unsigned int v9; // r14d
  __int64 v11; // rax
  int v12; // edx
  unsigned int v13; // r11d
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // r8d
  unsigned int v18; // r9d
  unsigned int v20; // r9d
  int v21; // eax
  int v22; // r15d
  unsigned int v23; // eax
  unsigned int v24; // eax
  unsigned int v25; // r8d
  unsigned int v26; // r9d
  char v27; // al
  __int64 v28; // rax
  int v29; // eax
  int v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  unsigned int v34; // [rsp+10h] [rbp-40h]
  unsigned int v35; // [rsp+10h] [rbp-40h]
  unsigned int v36; // [rsp+10h] [rbp-40h]
  unsigned int v37; // [rsp+18h] [rbp-38h]
  unsigned int v39; // [rsp+1Ch] [rbp-34h]

  *a7 = 0;
  v7 = *(_BYTE *)(a1 + 18);
  if ( (v7 & 6) == 0 )
    return 1;
  v9 = (a4 + 63) >> 6;
  if ( (v7 & 7) == 3 )
  {
    sub_16A7020(a2, 0, v9);
    v18 = 0;
    *a7 = ((*(_BYTE *)(a1 + 18) >> 3) ^ 1) & 1;
    return v18;
  }
  v11 = sub_16984A0(a1);
  v12 = *(__int16 *)(a1 + 16);
  if ( *(__int16 *)(a1 + 16) < 0 )
  {
    v33 = v11;
    sub_16A7020(a2, 0, v9);
    v16 = v33;
    v17 = *(_DWORD *)(*(_QWORD *)a1 + 4LL) - 1 - *(__int16 *)(a1 + 16);
  }
  else
  {
    v13 = v12 + 1;
    if ( a4 < v12 + 1 )
      return 1;
    v14 = *(unsigned int *)(*(_QWORD *)a1 + 4LL);
    if ( (unsigned int)v14 <= v13 )
    {
      sub_16A8750(a2, v9, v11, v14, 0);
      sub_16A7D00(a2);
      v20 = 0;
      goto LABEL_13;
    }
    v30 = v14 - v13;
    v32 = v11;
    sub_16A8750(a2, v9, v11, v13, (unsigned int)v14 - v13);
    v16 = v32;
    v17 = v30;
  }
  if ( v17 && (v35 = v17, v31 = v16, v37 = sub_1698310(a1), v24 = sub_16A7110(v31, v37), v25 = v35, v35 > v24) )
  {
    if ( v35 == v24 + 1 )
    {
      v26 = 2;
    }
    else if ( v35 > v37 << 6 || (v29 = sub_16A70B0(v31, v35 - 1), v25 = v35, v26 = 3, !v29) )
    {
      v26 = 1;
    }
    v36 = v26;
    v27 = sub_1698E10(a1, a6, v26, v25);
    v20 = v36;
    if ( v27 )
    {
      v28 = sub_16A73B0(a2, 1, v9);
      v20 = v36;
      if ( v28 )
        return 1;
    }
  }
  else
  {
    v20 = 0;
  }
LABEL_13:
  v34 = v20;
  v21 = sub_16A7150(a2, v9, v15);
  v18 = v34;
  v22 = v21;
  v23 = v21 + 1;
  if ( (*(_BYTE *)(a1 + 18) & 8) == 0 )
  {
    if ( (a5 ^ 1) + a4 > v23 )
      goto LABEL_17;
    return 1;
  }
  if ( a5 )
  {
    if ( a4 == v23 )
    {
      if ( v22 == (unsigned int)sub_16A7110(a2, v9) )
      {
        v18 = v34;
        goto LABEL_16;
      }
    }
    else if ( a4 >= v23 )
    {
      goto LABEL_16;
    }
    return 1;
  }
  if ( v23 )
    return 1;
LABEL_16:
  v39 = v18;
  sub_16A98A0(a2, v9);
  v18 = v39;
LABEL_17:
  if ( v18 )
    return 16;
  else
    *a7 = 1;
  return v18;
}
