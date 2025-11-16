// Function: sub_16995F0
// Address: 0x16995f0
//
__int64 __fastcall sub_16995F0(__int64 a1, __int16 *a2, unsigned int a3, bool *a4)
{
  __int16 *v5; // r15
  unsigned int v6; // r12d
  unsigned int v7; // eax
  unsigned int v8; // r12d
  int v9; // r13d
  unsigned int v10; // r10d
  char v11; // al
  __int64 v12; // r15
  char v13; // al
  unsigned int v14; // r12d
  void *v15; // rax
  char v16; // al
  __int64 v18; // rax
  char v19; // al
  __int64 v20; // r15
  unsigned int v21; // eax
  __int64 v22; // rdx
  unsigned int v23; // r10d
  char v24; // al
  __int64 v25; // rax
  int v26; // eax
  int v27; // esi
  int v28; // eax
  __int64 *v29; // rax
  _QWORD *v30; // rax
  int v31; // eax
  unsigned int v32; // [rsp+8h] [rbp-48h]
  unsigned int v33; // [rsp+8h] [rbp-48h]
  unsigned int v34; // [rsp+8h] [rbp-48h]
  unsigned int v35; // [rsp+8h] [rbp-48h]
  bool v38; // [rsp+1Bh] [rbp-35h]
  unsigned int v39; // [rsp+1Ch] [rbp-34h]
  unsigned int v40; // [rsp+1Ch] [rbp-34h]
  unsigned int v41; // [rsp+1Ch] [rbp-34h]

  v5 = *(__int16 **)a1;
  v6 = *((_DWORD *)a2 + 1) + 64;
  v7 = sub_1698310(a1);
  v8 = v6 >> 6;
  v9 = *((_DWORD *)a2 + 1) - *((_DWORD *)v5 + 1);
  v10 = v7;
  v38 = a2 != (__int16 *)&unk_42AE9B0 && v5 == (__int16 *)&unk_42AE9B0;
  if ( v38 )
  {
    if ( (*(_BYTE *)(a1 + 18) & 7) == 1 )
    {
      v41 = v7;
      v29 = (__int64 *)sub_1698470(a1);
      v10 = v41;
      if ( *v29 < 0 )
      {
        v30 = (_QWORD *)sub_1698470(a1);
        v10 = v41;
        v38 = ((*v30 >> 62) ^ 1) & 1;
      }
    }
    else
    {
      v38 = 0;
    }
  }
  v39 = 0;
  if ( v9 >= 0 )
  {
LABEL_3:
    if ( v10 >= v8 )
      goto LABEL_4;
LABEL_35:
    v34 = v10;
    v12 = sub_2207820(8LL * v8);
    sub_16A7020(v12, 0, v8);
    v24 = *(_BYTE *)(a1 + 18) & 7;
    if ( v24 != 1 && (!v24 || v24 == 3) )
      goto LABEL_10;
    v25 = sub_1698470(a1);
    sub_16A7050(v12, v25, v34);
    sub_16983A0(a1);
    *(_QWORD *)(a1 + 8) = v12;
    goto LABEL_11;
  }
  v19 = *(_BYTE *)(a1 + 18) & 7;
  if ( (*(_BYTE *)(a1 + 18) & 6) != 0 && v19 != 3 )
  {
    v35 = v10;
    v26 = sub_1698BA0(a1);
    v27 = *(__int16 *)(a1 + 16);
    v10 = v35;
    v28 = v26 - *((_DWORD *)v5 + 1) + 1;
    if ( v27 + v28 < a2[1] )
      v28 = a2[1] - v27;
    if ( v28 < v9 )
      v28 = v9;
    if ( v28 < 0 )
    {
      *(_WORD *)(a1 + 16) += v28;
      v9 -= v28;
      if ( v9 >= 0 )
        goto LABEL_3;
    }
    v19 = *(_BYTE *)(a1 + 18) & 7;
  }
  if ( v19 != 1 && (!v19 || v19 == 3) )
  {
    v39 = 0;
    goto LABEL_3;
  }
  v40 = v10;
  v20 = sub_1698470(a1);
  v32 = v40;
  v21 = sub_16A7110(v20, v40);
  v22 = (unsigned int)-v9;
  v39 = 0;
  v23 = v32;
  if ( -v9 > v21 )
  {
    if ( (_DWORD)v22 == v21 + 1 )
    {
      v39 = 2;
    }
    else if ( (unsigned int)v22 > v32 << 6
           || (v31 = sub_16A70B0(v20, (unsigned int)~v9), v39 = 3, v22 = (unsigned int)-v9, v23 = v32, !v31) )
    {
      v39 = 1;
    }
  }
  v33 = v23;
  sub_16A8050(v20, v23, v22);
  v10 = v33;
  if ( v33 < v8 )
    goto LABEL_35;
LABEL_4:
  if ( v8 == 1 && v10 != 1 )
  {
    v11 = *(_BYTE *)(a1 + 18) & 7;
    if ( v11 == 1 || v11 && v11 != 3 )
    {
      v12 = *(_QWORD *)sub_1698470(a1);
LABEL_10:
      sub_16983A0(a1);
      *(_QWORD *)(a1 + 8) = v12;
      goto LABEL_11;
    }
    sub_16983A0(a1);
    *(_QWORD *)(a1 + 8) = 0;
  }
LABEL_11:
  *(_QWORD *)a1 = a2;
  if ( v9 > 0 )
  {
    v13 = *(_BYTE *)(a1 + 18) & 7;
    if ( v13 != 1 && (!v13 || v13 == 3) )
    {
LABEL_15:
      v14 = 0;
      *a4 = 0;
      return v14;
    }
    v15 = (void *)sub_1698470(a1);
    sub_16A7D00(v15);
  }
  v16 = *(_BYTE *)(a1 + 18) & 7;
  if ( v16 != 1 )
  {
    if ( v16 != 3 && v16 )
    {
      v14 = sub_1698EC0((__int16 **)a1, a3, v39);
      *a4 = v14 != 0;
      return v14;
    }
    goto LABEL_15;
  }
  v14 = 0;
  *a4 = v38 || v39 != 0;
  if ( !v38 && *(_UNKNOWN **)a1 == &unk_42AE9B0 )
  {
    v18 = sub_1698470(a1);
    sub_16A70D0(v18, 63);
  }
  return v14;
}
