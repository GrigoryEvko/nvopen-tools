// Function: sub_E5C4E0
// Address: 0xe5c4e0
//
__int64 __fastcall sub_E5C4E0(__int64 a1, __int64 *a2, __int64 a3, _QWORD *a4, __int64 a5, _QWORD *a6, _BYTE *a7)
{
  __int64 v11; // rcx
  __int64 v12; // rdi
  unsigned int v13; // eax
  unsigned int v14; // r12d
  _DWORD *v15; // rax
  int v16; // r10d
  int v18; // r11d
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 (*v23)(); // rax
  __int64 v24; // rdi
  __int64 (*v25)(); // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 *v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  char v35; // [rsp+8h] [rbp-88h]
  char v36; // [rsp+8h] [rbp-88h]
  char v37; // [rsp+8h] [rbp-88h]
  char v38; // [rsp+Ch] [rbp-84h]
  char v39; // [rsp+Ch] [rbp-84h]
  char v40; // [rsp+Ch] [rbp-84h]
  int v41; // [rsp+Ch] [rbp-84h]
  int v42; // [rsp+Ch] [rbp-84h]
  int v43; // [rsp+Ch] [rbp-84h]
  __int64 v44; // [rsp+10h] [rbp-80h]
  int v45; // [rsp+10h] [rbp-80h]
  int v46; // [rsp+10h] [rbp-80h]
  int v47; // [rsp+10h] [rbp-80h]
  char v48; // [rsp+10h] [rbp-80h]
  __int64 *v49; // [rsp+10h] [rbp-80h]
  _QWORD v52[4]; // [rsp+30h] [rbp-60h] BYREF
  char v53; // [rsp+50h] [rbp-40h]
  char v54; // [rsp+51h] [rbp-3Fh]

  v11 = *(_QWORD *)a1;
  v12 = *a2;
  *a6 = 0;
  *a7 = 0;
  v44 = v11;
  v13 = sub_E81950(v12, a4, a1, a2);
  if ( !(_BYTE)v13 )
  {
    v26 = a2[2];
    v54 = 1;
    v53 = 3;
    v14 = 1;
    v52[0] = "expected relocatable expression";
    sub_E66880(v44, v26, v52);
    return v14;
  }
  v14 = v13;
  v15 = (_DWORD *)a4[1];
  if ( v15 && (*v15 & 0xFFFF00) != 0 )
  {
    v30 = a2[2];
    v54 = 1;
    v53 = 3;
    v52[0] = "unsupported subtraction of qualified symbol";
    sub_E66880(v44, v30, v52);
    return v14;
  }
  v16 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 64LL))(
                      *(_QWORD *)(a1 + 8),
                      *((unsigned int *)a2 + 3))
                  + 16);
  if ( (v16 & 4) != 0 )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, __int64 *, __int64, _QWORD *, __int64, _QWORD *, _BYTE *))(**(_QWORD **)(a1 + 8) + 96LL))(
             *(_QWORD *)(a1 + 8),
             a1,
             a2,
             a3,
             a4,
             a5,
             a6,
             a7);
  v18 = v16 & 1;
  if ( (v16 & 1) == 0 )
  {
    if ( !*a4 && !a4[1] )
      goto LABEL_10;
LABEL_9:
    v14 = 0;
    goto LABEL_10;
  }
  if ( a4[1] )
    goto LABEL_9;
  v27 = *a4;
  if ( !*a4 )
    goto LABEL_9;
  if ( (*(_DWORD *)v27 & 0xFFFF00) != 0 )
    goto LABEL_9;
  v28 = *(__int64 **)(v27 + 16);
  if ( !*v28 )
  {
    if ( (*((_BYTE *)v28 + 9) & 0x70) != 0x20 )
      goto LABEL_9;
    if ( *((char *)v28 + 8) < 0 )
      goto LABEL_9;
    *((_BYTE *)v28 + 8) |= 8u;
    v37 = v16;
    v43 = v16 & 1;
    v49 = v28;
    v34 = sub_E807D0(v28[3]);
    v28 = v49;
    v18 = v43;
    LOBYTE(v16) = v37;
    *v49 = v34;
    if ( !v34 )
      goto LABEL_9;
  }
  if ( (v16 & 8) == 0 )
  {
    v47 = v18;
    v40 = v16;
    v29 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64 *, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 24) + 40LL))(
            *(_QWORD *)(a1 + 24),
            a1,
            v28,
            a3,
            0,
            1);
    v18 = v47;
    LOBYTE(v16) = v40;
    v14 = v29;
  }
LABEL_10:
  *a6 = a4[2];
  if ( *a4 )
  {
    v19 = *(_QWORD *)(*a4 + 16LL);
    if ( *(_QWORD *)v19
      || (*(_BYTE *)(v19 + 9) & 0x70) == 0x20
      && *(char *)(v19 + 8) >= 0
      && (*(_BYTE *)(v19 + 8) |= 8u,
          v36 = v16,
          v42 = v18,
          v33 = sub_E807D0(*(_QWORD *)(v19 + 24)),
          v18 = v42,
          LOBYTE(v16) = v36,
          (*(_QWORD *)v19 = v33) != 0) )
    {
      v38 = v16;
      v45 = v18;
      *a6 += sub_E5C4C0(a1, v19);
      v18 = v45;
      LOBYTE(v16) = v38;
    }
  }
  v20 = a4[1];
  if ( v20 )
  {
    v21 = *(_QWORD *)(v20 + 16);
    if ( *(_QWORD *)v21
      || (*(_BYTE *)(v21 + 9) & 0x70) == 0x20
      && *(char *)(v21 + 8) >= 0
      && (*(_BYTE *)(v21 + 8) |= 8u,
          v35 = v16,
          v41 = v18,
          v32 = sub_E807D0(*(_QWORD *)(v21 + 24)),
          v18 = v41,
          LOBYTE(v16) = v35,
          (*(_QWORD *)v21 = v32) != 0) )
    {
      v39 = v16;
      v46 = v18;
      *a6 -= sub_E5C4C0(a1, v21);
      v18 = v46;
      LOBYTE(v16) = v39;
    }
  }
  if ( v18 )
  {
    v48 = v16;
    v31 = sub_E5C2C0(a1, a3) + *((unsigned int *)a2 + 2);
    if ( (v48 & 2) != 0 )
      v31 &= 0xFFFFFFFFFFFFFFFCLL;
    *a6 -= v31;
  }
  if ( (_BYTE)v14 )
  {
    v22 = *(_QWORD *)(a1 + 8);
    v23 = *(__int64 (**)())(*(_QWORD *)v22 + 72LL);
    if ( v23 == sub_E5B7F0
      || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, _QWORD *, _QWORD, __int64))v23)(
            v22,
            a1,
            a2,
            a4,
            *a6,
            a5) )
    {
      return v14;
    }
    *a7 = 1;
  }
  if ( !*a4 || !a4[1] || (*(_DWORD *)*a4 & 0xFFFF00) != 0 )
    return 0;
  v24 = *(_QWORD *)(a1 + 8);
  v14 = 0;
  v25 = *(__int64 (**)())(*(_QWORD *)v24 + 104LL);
  if ( v25 == sub_E5B820 )
    return v14;
  return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *, _QWORD *, _QWORD *))v25)(
           v24,
           a1,
           a3,
           a2,
           a4,
           a6);
}
