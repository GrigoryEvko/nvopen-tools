// Function: sub_9A3DF0
// Address: 0x9a3df0
//
__int64 __fastcall sub_9A3DF0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        unsigned int a4,
        unsigned __int8 *a5,
        unsigned __int8 *a6,
        unsigned __int8 a7,
        char a8)
{
  unsigned __int8 *v8; // r15
  __int64 v13; // r8
  unsigned int v14; // r11d
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // ebx
  char v23; // al
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  char v26; // al
  char v27; // al
  __int64 v28; // [rsp+0h] [rbp-B0h]
  unsigned int v30; // [rsp+18h] [rbp-98h]
  unsigned int v31; // [rsp+18h] [rbp-98h]
  unsigned int v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-88h]
  __int64 v35; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-78h]
  __int64 v37; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+48h] [rbp-68h]
  __int64 v39; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-58h]
  unsigned __int64 v41; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-48h]
  __int64 v43; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-38h]

  v8 = a6;
  if ( !a8 )
  {
    sub_9B0110(&v33, a5, a1, a2, a3);
    sub_9B0110(&v37, v8, a1, a2, a3);
    v14 = a4;
    if ( v34 > 0x40 )
      v15 = *(_QWORD *)(v33 + 8LL * ((v34 - 1) >> 6));
    else
      v15 = v33;
    if ( (v15 & (1LL << ((unsigned __int8)v34 - 1))) != 0 )
    {
      v16 = 1LL << ((unsigned __int8)v38 - 1);
      if ( v38 > 0x40 )
      {
        if ( (*(_QWORD *)(v37 + 8LL * ((v38 - 1) >> 6)) & v16) == 0 )
          goto LABEL_9;
      }
      else if ( (v37 & v16) == 0 )
      {
        goto LABEL_9;
      }
      if ( (unsigned __int8)sub_9A6530(v8, a1, a3, a2) )
        goto LABEL_19;
      v23 = sub_9A6530(a5, a1, a3, a2);
      v14 = a4;
      if ( v23 )
        goto LABEL_19;
    }
LABEL_9:
    v17 = v36;
    v18 = v35;
    if ( v36 > 0x40 )
      v18 = *(_QWORD *)(v35 + 8LL * ((v36 - 1) >> 6));
    if ( (v18 & (1LL << ((unsigned __int8)v36 - 1))) == 0 )
    {
LABEL_14:
      v20 = 1LL << ((unsigned __int8)v34 - 1);
      if ( v34 > 0x40 )
      {
        if ( (*(_QWORD *)(v33 + 8LL * ((v34 - 1) >> 6)) & v20) == 0 )
          goto LABEL_16;
      }
      else if ( (v33 & v20) == 0 )
      {
        goto LABEL_16;
      }
      if ( (unsigned __int8)sub_9A1DB0(v8, 0, a2, a3, v13) )
        goto LABEL_19;
LABEL_16:
      v21 = 1LL << ((unsigned __int8)v38 - 1);
      if ( v38 > 0x40 )
      {
        if ( (*(_QWORD *)(v37 + 8LL * ((v38 - 1) >> 6)) & v21) == 0 )
          goto LABEL_20;
      }
      else if ( (v37 & v21) == 0 )
      {
        goto LABEL_20;
      }
      if ( (unsigned __int8)sub_9A1DB0(a5, 0, a2, a3, v13) )
      {
LABEL_19:
        LODWORD(v8) = 1;
LABEL_26:
        if ( v40 > 0x40 && v39 )
          j_j___libc_free_0_0(v39);
        if ( v38 > 0x40 && v37 )
          j_j___libc_free_0_0(v37);
        if ( v36 > 0x40 && v35 )
          j_j___libc_free_0_0(v35);
        if ( v34 > 0x40 )
        {
          if ( v33 )
            j_j___libc_free_0_0(v33);
        }
        return (unsigned int)v8;
      }
LABEL_20:
      sub_C70430(&v41, 1, a7, 0, &v33, &v37);
      v22 = v44;
      if ( v44 <= 0x40 )
      {
        LOBYTE(v8) = v43 != 0;
      }
      else
      {
        LOBYTE(v8) = v22 != (unsigned int)sub_C444A0(&v43);
        if ( v43 )
          j_j___libc_free_0_0(v43);
      }
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
      goto LABEL_26;
    }
    v19 = 1LL << ((unsigned __int8)v40 - 1);
    if ( v40 > 0x40 )
    {
      if ( (*(_QWORD *)(v39 + 8LL * ((v40 - 1) >> 6)) & v19) == 0 )
        goto LABEL_14;
    }
    else if ( (v39 & v19) == 0 )
    {
      goto LABEL_14;
    }
    v42 = v14;
    v24 = ~(1LL << ((unsigned __int8)v14 - 1));
    if ( v14 > 0x40 )
    {
      v28 = ~(1LL << ((unsigned __int8)v14 - 1));
      v32 = v14 - 1;
      sub_C43690(&v41, -1, 1);
      v14 = v42;
      v24 = v28;
      if ( v42 > 0x40 )
      {
        *(_QWORD *)(v41 + 8LL * (v32 >> 6)) &= v28;
        v17 = v36;
        v14 = v42;
LABEL_58:
        if ( v17 <= 0x40 )
        {
          if ( (v41 & v35) != 0 )
            goto LABEL_69;
        }
        else
        {
          v30 = v14;
          v26 = sub_C446A0(&v35, &v41);
          v14 = v30;
          if ( v26 )
            goto LABEL_69;
        }
        if ( v40 <= 0x40 )
        {
          if ( (v41 & v39) == 0 )
            goto LABEL_62;
        }
        else
        {
          v31 = v14;
          v27 = sub_C446A0(&v39, &v41);
          v14 = v31;
          if ( !v27 )
          {
LABEL_62:
            if ( v14 > 0x40 && v41 )
              j_j___libc_free_0_0(v41);
            goto LABEL_14;
          }
        }
LABEL_69:
        if ( v14 > 0x40 && v41 )
          j_j___libc_free_0_0(v41);
        goto LABEL_19;
      }
      v17 = v36;
    }
    else
    {
      v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
      if ( !v14 )
        v25 = 0;
      v41 = v25;
    }
    v41 &= v24;
    goto LABEL_58;
  }
  LODWORD(v8) = sub_9A6530(a6, a1, a3, a2);
  if ( (_BYTE)v8 )
    return (unsigned int)v8;
  return sub_9A6530(a5, a1, a3, a2);
}
