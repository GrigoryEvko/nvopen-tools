// Function: sub_390B240
// Address: 0x390b240
//
__int64 __fastcall sub_390B240(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6, _BYTE *a7)
{
  __int64 v10; // rdi
  __int64 v11; // r14
  unsigned int v12; // eax
  unsigned int v13; // r15d
  __int64 v14; // rax
  int v15; // r14d
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  __int64 v23; // rax
  __int64 v24; // r15
  unsigned __int64 v25; // rax
  __int64 v26; // rdi
  int v27; // r14d
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rdx
  _QWORD v36[2]; // [rsp+20h] [rbp-50h] BYREF
  char v37; // [rsp+30h] [rbp-40h]
  char v38; // [rsp+31h] [rbp-3Fh]

  v10 = *(_QWORD *)a3;
  v11 = *(_QWORD *)a1;
  *a6 = 0;
  *a7 = 0;
  v12 = sub_38CF2C0(v10, (__int64)a5, a2, a3);
  if ( !(_BYTE)v12 )
  {
    v29 = *(_QWORD *)(a3 + 16);
    v38 = 1;
    v37 = 3;
    v13 = 1;
    v36[0] = "expected relocatable expression";
    sub_38BE3D0(v11, v29, (__int64)v36);
    return v13;
  }
  v13 = v12;
  v14 = a5[1];
  if ( v14 && *(_WORD *)(v14 + 16) )
  {
    v30 = *(_QWORD *)(a3 + 16);
    v38 = 1;
    v37 = 3;
    v36[0] = "unsupported subtraction of qualified symbol";
    sub_38BE3D0(v11, v30, (__int64)v36);
    return v13;
  }
  v15 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(
                      *(_QWORD *)(a1 + 8),
                      *(unsigned int *)(a3 + 12))
                  + 16)
      & 1;
  if ( !v15 )
  {
    if ( !*a5 && !a5[1] )
      goto LABEL_7;
    goto LABEL_6;
  }
  if ( a5[1]
    || (v23 = *a5) == 0
    || *(_WORD *)(v23 + 16)
    || (v24 = *(_QWORD *)(v23 + 24), (*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL) == 0)
    && ((*(_BYTE *)(v24 + 9) & 0xC) != 8
     || (*(_BYTE *)(v24 + 8) |= 4u,
         v25 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v24 + 24)),
         *(_QWORD *)v24 = v25 | *(_QWORD *)v24 & 7LL,
         !v25))
    || (v26 = *(_QWORD *)(a1 + 24)) == 0 )
  {
LABEL_6:
    v13 = 0;
    goto LABEL_7;
  }
  v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v26 + 48LL))(
          v26,
          a1,
          v24,
          a4,
          0,
          1);
LABEL_7:
  *a6 = a5[2];
  if ( *a5 )
  {
    v16 = *(_QWORD *)(*a5 + 24LL);
    if ( (*(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL) != 0
      || (*(_BYTE *)(v16 + 9) & 0xC) == 8
      && (*(_BYTE *)(v16 + 8) |= 4u,
          v32 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v16 + 24)),
          *(_QWORD *)v16 = v32 | *(_QWORD *)v16 & 7LL,
          v32) )
    {
      *a6 += sub_38D0440(a2, v16);
    }
  }
  v17 = a5[1];
  if ( v17 )
  {
    v18 = *(_QWORD *)(v17 + 24);
    if ( (*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL) != 0
      || (*(_BYTE *)(v18 + 9) & 0xC) == 8
      && (*(_BYTE *)(v18 + 8) |= 4u,
          v31 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24)),
          *(_QWORD *)v18 = v31 | *(_QWORD *)v18 & 7LL,
          v31) )
    {
      *a6 -= sub_38D0440(a2, v18);
    }
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(
          *(_QWORD *)(a1 + 8),
          *(unsigned int *)(a3 + 12));
  if ( v15 )
  {
    v27 = *(_DWORD *)(v19 + 16);
    v28 = *(_DWORD *)(a3 + 8) + (unsigned int)sub_38D01B0((__int64)a2, a4);
    if ( (v27 & 2) != 0 )
      v28 = (unsigned int)v28 & 0xFFFFFFFC;
    *a6 -= v28;
  }
  if ( (_BYTE)v13 )
  {
    v20 = *(_QWORD *)(a1 + 8);
    v21 = *(__int64 (**)())(*(_QWORD *)v20 + 56LL);
    if ( v21 != sub_390A060 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD *))v21)(v20, a1, a3, a5) )
      {
        v13 = 0;
        *a7 = 1;
      }
    }
  }
  return v13;
}
