// Function: sub_23D0CA0
// Address: 0x23d0ca0
//
__int64 __fastcall sub_23D0CA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  char v6; // bl
  __int64 v7; // r13
  __int64 v8; // rcx
  unsigned int v9; // ebx
  bool v10; // al
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int8 *v14; // r13
  __int64 (__fastcall *v15)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v16; // rsi
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 **v19; // rbx
  __int64 (__fastcall *v20)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v21; // r13
  _BYTE *v22; // rax
  __int64 v23; // rdx
  unsigned int *v24; // r13
  __int64 v25; // rdx
  _QWORD *v26; // rax
  unsigned int *v27; // rbx
  unsigned int *v28; // r14
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // rbx
  __int64 v32; // rdx
  _BYTE *v33; // rax
  unsigned int v34; // ebx
  int v35; // r14d
  unsigned int v36; // r15d
  bool v37; // bl
  __int64 v38; // rax
  unsigned int v39; // ebx
  unsigned int *v40; // [rsp-158h] [rbp-158h]
  unsigned __int8 *v41; // [rsp-150h] [rbp-150h]
  unsigned __int8 v42; // [rsp-150h] [rbp-150h]
  unsigned __int8 *v43; // [rsp-148h] [rbp-148h] BYREF
  unsigned __int64 v44; // [rsp-140h] [rbp-140h] BYREF
  unsigned int v45; // [rsp-138h] [rbp-138h]
  char v46; // [rsp-130h] [rbp-130h]
  char v47; // [rsp-12Fh] [rbp-12Fh]
  _BYTE v48[32]; // [rsp-128h] [rbp-128h] BYREF
  __int16 v49; // [rsp-108h] [rbp-108h]
  _BYTE v50[32]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v51; // [rsp-D8h] [rbp-D8h]
  unsigned int *v52; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned int v53; // [rsp-C0h] [rbp-C0h]
  __int64 v54; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v55; // [rsp-90h] [rbp-90h]
  __int64 v56; // [rsp-88h] [rbp-88h]
  __int64 v57; // [rsp-78h] [rbp-78h]
  __int64 v58; // [rsp-70h] [rbp-70h]
  void *v59; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)a1 != 57 )
    return 0;
  v2 = *(_QWORD *)(a1 - 64);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 )
  {
    v4 = *(_QWORD *)(a1 - 32);
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 && !*(_QWORD *)(v5 + 8) && *(_BYTE *)v4 == 57 )
      goto LABEL_7;
    return 0;
  }
  if ( *(_QWORD *)(v3 + 8) )
  {
    v7 = *(_QWORD *)(a1 - 32);
    v8 = *(_QWORD *)(v7 + 16);
    if ( !v8 )
      return 0;
  }
  else
  {
    if ( *(_BYTE *)v2 == 57 )
    {
LABEL_7:
      v6 = 1;
      goto LABEL_22;
    }
    v7 = *(_QWORD *)(a1 - 32);
    v8 = *(_QWORD *)(v7 + 16);
    if ( !v8 )
      goto LABEL_16;
  }
  if ( !*(_QWORD *)(v8 + 8) && *(_BYTE *)v7 == 57 )
    goto LABEL_7;
  if ( *(_QWORD *)(v3 + 8) )
    return 0;
LABEL_16:
  if ( *(_BYTE *)v2 != 58 )
    return 0;
  if ( *(_BYTE *)v7 == 17 )
  {
    v9 = *(_DWORD *)(v7 + 32);
    if ( v9 <= 0x40 )
      v10 = *(_QWORD *)(v7 + 24) == 1;
    else
      v10 = v9 - 1 == (unsigned int)sub_C444A0(v7 + 24);
  }
  else
  {
    v31 = *(_QWORD *)(v7 + 8);
    v32 = (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17;
    if ( (unsigned int)v32 > 1 || *(_BYTE *)v7 > 0x15u )
      return 0;
    v33 = sub_AD7630(v7, 0, v32);
    if ( !v33 || *v33 != 17 )
    {
      if ( *(_BYTE *)(v31 + 8) != 17 )
        return 0;
      v35 = *(_DWORD *)(v31 + 32);
      v36 = 0;
      v37 = 0;
      while ( v35 != v36 )
      {
        v38 = sub_AD69F0((unsigned __int8 *)v7, v36);
        if ( !v38 )
          return 0;
        if ( *(_BYTE *)v38 != 13 )
        {
          if ( *(_BYTE *)v38 != 17 )
            return 0;
          v39 = *(_DWORD *)(v38 + 32);
          v37 = v39 <= 0x40 ? *(_QWORD *)(v38 + 24) == 1 : v39 - 1 == (unsigned int)sub_C444A0(v38 + 24);
          if ( !v37 )
            return 0;
        }
        ++v36;
      }
      if ( !v37 )
        return 0;
      goto LABEL_21;
    }
    v34 = *((_DWORD *)v33 + 8);
    if ( v34 <= 0x40 )
      v10 = *((_QWORD *)v33 + 3) == 1;
    else
      v10 = v34 - 1 == (unsigned int)sub_C444A0((__int64)(v33 + 24));
  }
  if ( !v10 )
    return 0;
LABEL_21:
  v6 = 0;
LABEL_22:
  v11 = sub_BCB060(*(_QWORD *)(a1 + 8));
  v43 = 0;
  v45 = v11;
  if ( v11 > 0x40 )
    sub_C43690((__int64)&v44, 0, 0);
  else
    v44 = 0;
  v46 = v6;
  v47 = 0;
  if ( v6 )
  {
    if ( sub_23CF510((_BYTE *)a1, (__int64)&v43, v12) && v47 )
    {
LABEL_26:
      sub_23D0AB0((__int64)&v52, a1, 0, 0, 0);
      v13 = sub_AD8D80(*(_QWORD *)(a1 + 8), (__int64)&v44);
      v14 = v43;
      v49 = 257;
      v41 = (unsigned __int8 *)v13;
      v15 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v57 + 16LL);
      if ( v15 == sub_9202E0 )
      {
        if ( *v43 > 0x15u || *v41 > 0x15u )
        {
LABEL_51:
          v51 = 257;
          v17 = sub_B504D0(28, (__int64)v14, (__int64)v41, (__int64)v50, 0, 0);
          v16 = v17;
          (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
            v58,
            v17,
            v48,
            v55,
            v56);
          v23 = 4LL * v53;
          v24 = v52;
          v40 = &v52[v23];
          if ( v24 != &v24[v23] )
          {
            do
            {
              v25 = *((_QWORD *)v24 + 1);
              v16 = *v24;
              v24 += 4;
              sub_B99FD0(v17, v16, v25);
            }
            while ( v40 != v24 );
          }
LABEL_32:
          v51 = 257;
          if ( v6 )
          {
            v18 = sub_92B530(&v52, 0x20u, v17, v41, (__int64)v50);
          }
          else
          {
            v22 = (_BYTE *)sub_AD6530(*(_QWORD *)(v17 + 8), v16);
            v18 = sub_92B530(&v52, 0x21u, v17, v22, (__int64)v50);
          }
          v19 = *(__int64 ***)(a1 + 8);
          v49 = 257;
          if ( v19 == *(__int64 ***)(v18 + 8) )
          {
            v21 = v18;
            goto LABEL_40;
          }
          v20 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v57 + 120LL);
          if ( v20 == sub_920130 )
          {
            if ( *(_BYTE *)v18 > 0x15u )
              goto LABEL_54;
            if ( (unsigned __int8)sub_AC4810(0x27u) )
              v21 = sub_ADAB70(39, v18, v19, 0);
            else
              v21 = sub_AA93C0(0x27u, v18, (__int64)v19);
          }
          else
          {
            v21 = v20(v57, 39u, (_BYTE *)v18, (__int64)v19);
          }
          if ( v21 )
          {
LABEL_40:
            sub_BD84D0(a1, v21);
            nullsub_61();
            v59 = &unk_49DA100;
            nullsub_63();
            if ( v52 != (unsigned int *)&v54 )
              _libc_free((unsigned __int64)v52);
            result = 1;
            goto LABEL_43;
          }
LABEL_54:
          v51 = 257;
          v26 = sub_BD2C40(72, unk_3F10A14);
          v21 = (__int64)v26;
          if ( v26 )
            sub_B515B0((__int64)v26, v18, (__int64)v19, (__int64)v50, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
            v58,
            v21,
            v48,
            v55,
            v56);
          v27 = v52;
          v28 = &v52[4 * v53];
          if ( v52 != v28 )
          {
            do
            {
              v29 = *((_QWORD *)v27 + 1);
              v30 = *v27;
              v27 += 4;
              sub_B99FD0(v21, v30, v29);
            }
            while ( v28 != v27 );
          }
          goto LABEL_40;
        }
        v16 = (__int64)v43;
        if ( (unsigned __int8)sub_AC47B0(28) )
          v17 = sub_AD5570(28, (__int64)v14, v41, 0, 0);
        else
          v17 = sub_AABE40(0x1Cu, v14, v41);
      }
      else
      {
        v16 = 28;
        v17 = v15(v57, 28u, v43, v41);
      }
      if ( v17 )
        goto LABEL_32;
      goto LABEL_51;
    }
  }
  else if ( sub_23CF510(*(_BYTE **)(a1 - 64), (__int64)&v43, v12) )
  {
    goto LABEL_26;
  }
  result = 0;
LABEL_43:
  if ( v45 > 0x40 )
  {
    if ( v44 )
    {
      v42 = result;
      j_j___libc_free_0_0(v44);
      return v42;
    }
  }
  return result;
}
