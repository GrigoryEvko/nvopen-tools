// Function: sub_13A83F0
// Address: 0x13a83f0
//
__int64 __fastcall sub_13A83F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // r10d
  __int64 v18; // r15
  __int64 v20; // rax
  char v21; // r14
  char v22; // r12
  char v23; // al
  char v24; // si
  char v25; // al
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rax
  bool v29; // r8
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+10h] [rbp-80h]
  char v33; // [rsp+10h] [rbp-80h]
  unsigned int v34; // [rsp+18h] [rbp-78h]
  __int64 v35; // [rsp+18h] [rbp-78h]
  char v36; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v39; // [rsp+28h] [rbp-68h]
  __int64 v40; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v41; // [rsp+38h] [rbp-58h]
  __int64 v42; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v43; // [rsp+48h] [rbp-48h]
  __int64 v44; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v45; // [rsp+58h] [rbp-38h]

  v34 = a6 - 1;
  v10 = sub_14806B0(*(_QWORD *)(a1 + 8), a3, a4, 0, 0);
  v11 = sub_1456040(v10);
  v12 = sub_13A7AF0(a1, a5, v11);
  if ( !v12 )
    goto LABEL_48;
  v13 = v12;
  v32 = v10;
  if ( !(unsigned __int8)sub_1477BC0(*(_QWORD *)(a1 + 8), v10) )
    v32 = sub_1480620(*(_QWORD *)(a1 + 8), v10, 0);
  v14 = sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  v15 = a2;
  if ( !v14 )
    v15 = sub_1480620(*(_QWORD *)(a1 + 8), a2, 0);
  v16 = sub_13A5B60(*(_QWORD *)(a1 + 8), v13, v15, 0, 0);
  v17 = sub_13A7760(a1, 38, v32, v16);
  if ( !(_BYTE)v17 )
  {
LABEL_48:
    if ( *(_WORD *)(v10 + 24) || *(_WORD *)(a2 + 24) )
    {
      v18 = 16LL * v34;
      if ( (unsigned __int8)sub_14560B0(v10) )
      {
        *(_QWORD *)(*(_QWORD *)(a7 + 48) + v18 + 8) = v10;
        sub_13A6300(a8, v10, a5);
        v17 = 0;
        *(_BYTE *)(*(_QWORD *)(a7 + 48) + 16LL * v34) &= 0xFAu;
      }
      else
      {
        if ( (unsigned __int8)sub_1456110(a2) )
        {
          *(_QWORD *)(*(_QWORD *)(a7 + 48) + v18 + 8) = v10;
          sub_13A6300(a8, v10, a5);
        }
        else
        {
          *(_BYTE *)(a7 + 43) = 0;
          v35 = sub_1480620(*(_QWORD *)(a1 + 8), v10, 0);
          v20 = sub_1480620(*(_QWORD *)(a1 + 8), a2, 0);
          sub_13A62E0(a8, a2, v20, v35, a5);
        }
        v33 = sub_1477CE0(*(_QWORD *)(a1 + 8), v10);
        v21 = sub_1477A90(*(_QWORD *)(a1 + 8), v10);
        v36 = sub_1477BC0(*(_QWORD *)(a1 + 8), v10);
        v22 = sub_1477A90(*(_QWORD *)(a1 + 8), a2);
        v23 = sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
        v17 = 0;
        v24 = v23;
        if ( v21 || (v25 = 1, v22) )
          v25 = (v24 | v36) ^ 1;
        if ( !v33 )
          v25 |= 2u;
        if ( !v36 && !v22 || !v21 && !v24 )
          v25 |= 4u;
        *(_BYTE *)(*(_QWORD *)(a7 + 48) + v18) = *(_BYTE *)(*(_QWORD *)(a7 + 48) + v18) & 7 & v25
                                               | *(_BYTE *)(*(_QWORD *)(a7 + 48) + v18) & 0xF8;
      }
      return v17;
    }
    v26 = *(_QWORD *)(v10 + 32);
    v39 = *(_DWORD *)(v26 + 32);
    if ( v39 > 0x40 )
      sub_16A4FD0(&v38, v26 + 24);
    else
      v38 = *(_QWORD *)(v26 + 24);
    v27 = *(_QWORD *)(a2 + 32);
    v41 = *(_DWORD *)(v27 + 32);
    if ( v41 > 0x40 )
      sub_16A4FD0(&v40, v27 + 24);
    else
      v40 = *(_QWORD *)(v27 + 24);
    v43 = v39;
    if ( v39 > 0x40 )
    {
      sub_16A4FD0(&v42, &v38);
      v45 = v39;
      if ( v39 > 0x40 )
      {
        sub_16A4FD0(&v44, &v38);
LABEL_29:
        sub_16AE5C0(&v38, &v40, &v42, &v44);
        if ( sub_13A38F0((__int64)&v44, 0) )
        {
          v31 = 16LL * v34;
          v37 = *(_QWORD *)(a7 + 48) + v31;
          *(_QWORD *)(v37 + 8) = sub_145CF40(*(_QWORD *)(a1 + 8), &v42);
          v28 = sub_145CF40(*(_QWORD *)(a1 + 8), &v42);
          sub_13A6300(a8, v28, a5);
          if ( sub_13A39D0((__int64)&v42, 0) )
          {
            *(_BYTE *)(*(_QWORD *)(a7 + 48) + v31) &= 0xF9u;
          }
          else
          {
            v29 = sub_13A3940((__int64)&v42, 0);
            v30 = *(_QWORD *)(a7 + 48);
            if ( v29 )
              *(_BYTE *)(v30 + v31) &= 0xFCu;
            else
              *(_BYTE *)(v30 + v31) &= 0xFAu;
          }
          sub_135E100(&v44);
          sub_135E100(&v42);
          sub_135E100(&v40);
          sub_135E100(&v38);
          return 0;
        }
        else
        {
          sub_135E100(&v44);
          sub_135E100(&v42);
          sub_135E100(&v40);
          sub_135E100(&v38);
          return 1;
        }
      }
    }
    else
    {
      v45 = v39;
      v42 = v38;
    }
    v44 = v38;
    goto LABEL_29;
  }
  return v17;
}
