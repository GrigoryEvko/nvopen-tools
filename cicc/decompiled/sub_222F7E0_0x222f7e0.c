// Function: sub_222F7E0
// Address: 0x222f7e0
//
_QWORD *__fastcall sub_222F7E0(
        __int64 a1,
        __int64 a2,
        int a3,
        _QWORD *a4,
        int a5,
        int *a6,
        signed int a7,
        int a8,
        unsigned __int64 a9,
        __int64 a10,
        _DWORD *a11)
{
  _QWORD *v12; // r12
  unsigned int v14; // ebx
  __int64 v15; // rax
  unsigned __int64 v16; // r10
  _QWORD *v17; // r9
  __int64 v18; // rcx
  bool v19; // r8
  int v20; // r13d
  unsigned __int64 i; // r15
  char v22; // al
  unsigned __int8 v23; // al
  unsigned __int64 v24; // rax
  bool v25; // dl
  char v26; // bp
  int v27; // ebp
  char v28; // si
  __int64 (__fastcall *v29)(__int64, unsigned int); // rax
  int v31; // eax
  char *v32; // rax
  char v33; // al
  int v34; // eax
  int v35; // eax
  _QWORD *v37; // [rsp+0h] [rbp-68h]
  __int64 v38; // [rsp+0h] [rbp-68h]
  _QWORD *v39; // [rsp+0h] [rbp-68h]
  _QWORD *v40; // [rsp+0h] [rbp-68h]
  bool v41; // [rsp+8h] [rbp-60h]
  _QWORD *v42; // [rsp+8h] [rbp-60h]
  __int64 v43; // [rsp+8h] [rbp-60h]
  _QWORD *v44; // [rsp+8h] [rbp-60h]
  __int64 v45; // [rsp+8h] [rbp-60h]
  __int64 v46; // [rsp+10h] [rbp-58h]
  bool v47; // [rsp+10h] [rbp-58h]
  bool v48; // [rsp+10h] [rbp-58h]
  __int64 v49; // [rsp+10h] [rbp-58h]
  bool v50; // [rsp+10h] [rbp-58h]
  bool v52; // [rsp+2Fh] [rbp-39h]
  bool v53; // [rsp+2Fh] [rbp-39h]
  unsigned __int64 v54; // [rsp+80h] [rbp+18h]
  unsigned __int64 v55; // [rsp+80h] [rbp+18h]
  unsigned __int64 v56; // [rsp+80h] [rbp+18h]
  unsigned __int64 v57; // [rsp+80h] [rbp+18h]
  unsigned __int64 v58; // [rsp+80h] [rbp+18h]

  v12 = (_QWORD *)a2;
  v14 = 10;
  v15 = sub_222F790((_QWORD *)(a10 + 208), a2);
  v16 = a9;
  v17 = a4;
  v18 = v15;
  if ( a9 != 2 )
  {
    v14 = 1000;
    if ( a9 != 4 )
      v14 = 1;
  }
  v19 = a5 == -1;
  v20 = 0;
  for ( i = 0; ; ++i )
  {
    v25 = a3 == -1;
    v26 = v25 && v12 != 0;
    if ( v26 )
    {
      if ( v12[2] >= v12[3] )
      {
        v57 = v16;
        v53 = v19;
        v49 = v18;
        v44 = v17;
        v34 = (*(__int64 (__fastcall **)(_QWORD *))(*v12 + 72LL))(v12);
        v25 = a3 == -1;
        v17 = v44;
        v18 = v49;
        v19 = v53;
        if ( v34 != -1 )
          v26 = 0;
        v16 = v57;
        if ( v34 == -1 )
          v12 = 0;
      }
      else
      {
        v26 = 0;
      }
    }
    else
    {
      v26 = a3 == -1;
    }
    if ( v17 && v19 )
    {
      if ( v17[2] >= v17[3] )
      {
        v54 = v16;
        v52 = v19;
        v46 = v18;
        v41 = v25;
        v37 = v17;
        v31 = (*(__int64 (__fastcall **)(_QWORD *))(*v17 + 72LL))(v17);
        v17 = v37;
        v25 = v41;
        v18 = v46;
        v19 = v52;
        v16 = v54;
        if ( v31 == -1 )
        {
          v26 ^= 1u;
          v17 = 0;
        }
      }
    }
    else
    {
      v26 ^= v19;
    }
    if ( i >= v16 || !v26 )
      break;
    if ( v12 && v25 )
    {
      v32 = (char *)v12[2];
      if ( (unsigned __int64)v32 >= v12[3] )
      {
        v58 = v16;
        v50 = v19;
        v45 = v18;
        v40 = v17;
        v35 = (*(__int64 (__fastcall **)(_QWORD *))(*v12 + 72LL))(v12);
        v17 = v40;
        v18 = v45;
        v19 = v50;
        v16 = v58;
        v28 = v35;
        if ( v35 == -1 )
        {
          v28 = -1;
          v27 = 255;
          v12 = 0;
        }
        else
        {
          v27 = (unsigned __int8)v35;
        }
      }
      else
      {
        v27 = (unsigned __int8)*v32;
        v28 = *v32;
      }
    }
    else
    {
      v27 = (unsigned __int8)a3;
      v28 = a3;
    }
    v22 = *(_BYTE *)(v18 + v27 + 313);
    if ( !v22 )
    {
      v29 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v18 + 64LL);
      if ( v29 != sub_2216C50 )
      {
        v38 = v18;
        v55 = v16;
        v47 = v19;
        v42 = v17;
        v33 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v29)(v18, (unsigned int)v28, 42);
        v16 = v55;
        v18 = v38;
        v19 = v47;
        v17 = v42;
        v28 = v33;
      }
      if ( v28 == 42 )
        goto LABEL_26;
      *(_BYTE *)(v18 + v27 + 313) = v28;
      v22 = v28;
    }
    v23 = v22 - 48;
    if ( v23 > 9u )
      goto LABEL_26;
    v20 = (char)v23 + 10 * v20;
    if ( (int)(v20 * v14) > a8 || (int)(v14 + v20 * v14) <= a7 )
      goto LABEL_26;
    v24 = v12[2];
    v14 /= 0xAu;
    if ( v24 >= v12[3] )
    {
      v56 = v16;
      v48 = v19;
      v43 = v18;
      v39 = v17;
      (*(void (__fastcall **)(_QWORD *))(*v12 + 80LL))(v12);
      v16 = v56;
      v19 = v48;
      v18 = v43;
      v17 = v39;
    }
    else
    {
      v12[2] = v24 + 1;
    }
    a3 = -1;
  }
  if ( i == v16 )
  {
LABEL_29:
    *a6 = v20;
    return v12;
  }
LABEL_26:
  if ( v16 == 4 && i == 2 )
  {
    v20 -= 100;
    goto LABEL_29;
  }
  *a11 |= 4u;
  return v12;
}
