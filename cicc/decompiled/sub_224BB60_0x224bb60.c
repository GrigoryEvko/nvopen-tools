// Function: sub_224BB60
// Address: 0x224bb60
//
_QWORD *__fastcall sub_224BB60(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        int *a7,
        _BYTE *a8)
{
  int v8; // edx
  __int64 v10; // rax
  __int128 v11; // rax
  _QWORD *v12; // r12
  bool v13; // bp
  bool v14; // r8
  int v15; // eax
  unsigned __int64 v16; // r13
  bool v17; // r14
  bool v18; // bl
  char v19; // r15
  bool v20; // si
  unsigned __int64 v21; // rax
  bool v22; // di
  char v23; // r15
  char v24; // si
  char v25; // al
  char v26; // si
  int v27; // eax
  int *v28; // rax
  int v29; // edi
  int *v30; // rax
  int v31; // eax
  int *v32; // rax
  int v33; // eax
  int v34; // eax
  bool v35; // [rsp+Eh] [rbp-7Ah]
  bool v36; // [rsp+Eh] [rbp-7Ah]
  bool v37; // [rsp+Eh] [rbp-7Ah]
  _QWORD *v38; // [rsp+20h] [rbp-68h] BYREF
  __int64 v39; // [rsp+28h] [rbp-60h]
  _QWORD *v40; // [rsp+30h] [rbp-58h] BYREF
  __int64 v41; // [rsp+38h] [rbp-50h]
  unsigned __int64 v42[8]; // [rsp+48h] [rbp-40h] BYREF

  v40 = a2;
  v41 = a3;
  v38 = a4;
  v39 = a5;
  if ( (*(_BYTE *)(a6 + 24) & 1) == 0 )
  {
    v42[0] = -1;
    v40 = sub_224B080(a1, a2, a3, a4, a5, a6, a7, v42);
    LODWORD(v41) = v8;
    if ( v42[0] > 1 )
    {
      *a8 = 1;
      *a7 = 4;
      if ( sub_2247850((__int64)&v40, (__int64)&v38) )
        *a7 |= 2u;
    }
    else
    {
      *a8 = v42[0];
      *a8 &= 1u;
    }
    return v40;
  }
  v10 = sub_22462F0((__int64)v42, (__int64 *)(a6 + 208));
  *((_QWORD *)&v11 + 1) = *(_QWORD *)(v10 + 64);
  v12 = (_QWORD *)v10;
  *(_QWORD *)&v11 = *(_QWORD *)(v10 + 48);
  v13 = *((_QWORD *)&v11 + 1) == 0;
  v14 = (_QWORD)v11 == 0;
  if ( v11 == 0 )
    goto LABEL_55;
  v15 = v41;
  v16 = 0;
  v17 = 1;
  v18 = 1;
  while ( 1 )
  {
    v23 = v15 == -1;
    v24 = v23 & (v40 != 0);
    if ( v24 )
    {
      v30 = (int *)v40[2];
      if ( (unsigned __int64)v30 >= v40[3] )
      {
        v36 = v14;
        v31 = (*(__int64 (**)(void))(*v40 + 72LL))();
        v14 = v36;
      }
      else
      {
        v31 = *v30;
      }
      v23 = 0;
      if ( v31 == -1 )
      {
        v40 = 0;
        v23 = v24;
      }
    }
    v25 = (_DWORD)v39 == -1;
    v26 = v25 & (v38 != 0);
    if ( v26 )
    {
      v28 = (int *)v38[2];
      if ( (unsigned __int64)v28 >= v38[3] )
      {
        v35 = v14;
        v34 = (*(__int64 (**)(void))(*v38 + 72LL))();
        v14 = v35;
        v29 = v34;
      }
      else
      {
        v29 = *v28;
      }
      v25 = 0;
      if ( v29 == -1 )
        break;
    }
    if ( v23 == v25 )
      goto LABEL_35;
LABEL_20:
    v27 = v41;
    if ( (_DWORD)v41 == -1 && v40 )
    {
      v32 = (int *)v40[2];
      if ( (unsigned __int64)v32 >= v40[3] )
      {
        v37 = v14;
        v27 = (*(__int64 (**)(void))(*v40 + 72LL))();
        v14 = v37;
      }
      else
      {
        v27 = *v32;
      }
      if ( v27 == -1 )
        v40 = 0;
    }
    if ( !v13 )
      v18 = *(_DWORD *)(v12[7] + 4 * v16) == v27;
    if ( !v18 )
    {
      if ( v14 )
      {
        if ( v17 )
        {
          if ( v12[6] == v16 && v16 )
          {
            *a8 = 1;
            v33 = 0;
LABEL_67:
            *a7 = v33;
            return v40;
          }
LABEL_55:
          *a8 = 0;
          goto LABEL_56;
        }
LABEL_57:
        *a8 = 0;
        *a7 = 4;
        return v40;
      }
LABEL_8:
      v17 = *(_DWORD *)(v12[5] + 4 * v16) == v27;
      goto LABEL_9;
    }
    if ( !v14 )
      goto LABEL_8;
LABEL_9:
    v19 = v13 && !v17;
    if ( v19 )
    {
      if ( v18 )
      {
        if ( v12[8] == v16 && v16 )
        {
          *a8 = 0;
          v33 = 0;
          goto LABEL_67;
        }
        goto LABEL_55;
      }
      goto LABEL_57;
    }
    v20 = v17 || v18;
    if ( !v17 && !v18 )
      goto LABEL_57;
    ++v16;
    v21 = v40[2];
    if ( v21 >= v40[3] )
    {
      (*(void (__fastcall **)(_QWORD *))(*v40 + 80LL))(v40);
      v20 = v17 || v18;
    }
    else
    {
      v40[2] = v21 + 4;
    }
    LODWORD(v41) = -1;
    if ( !v18 )
    {
      v13 = v20;
LABEL_30:
      v17 = v20;
      v14 = v12[6] <= v16;
      v22 = v13 && v14;
      goto LABEL_16;
    }
    v13 = v12[8] <= v16;
    if ( v17 )
      goto LABEL_30;
    v22 = v12[8] <= v16;
    v14 = v18;
LABEL_16:
    v15 = -1;
    if ( v22 )
      goto LABEL_36;
  }
  v38 = 0;
  if ( v23 != v26 )
    goto LABEL_20;
LABEL_35:
  v19 = 1;
LABEL_36:
  if ( v18 && v12[8] == v16 && v16 )
  {
    *a8 = 0;
    if ( !v17 || v12[6] != v16 )
      goto LABEL_70;
LABEL_56:
    *a7 = 4;
    return v40;
  }
  if ( v17 && v12[6] == v16 && v16 )
  {
    *a8 = 1;
LABEL_70:
    v33 = 2 * (v19 != 0);
    goto LABEL_67;
  }
  *a8 = 0;
  if ( !v19 )
    goto LABEL_56;
  *a7 = 6;
  return v40;
}
