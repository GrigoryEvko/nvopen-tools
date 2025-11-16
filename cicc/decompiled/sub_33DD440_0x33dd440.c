// Function: sub_33DD440
// Address: 0x33dd440
//
__int64 __fastcall sub_33DD440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r15d
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned int v15; // eax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  unsigned int v19; // ebx
  _QWORD *v20; // r12
  unsigned int v21; // [rsp+4h] [rbp-CCh]
  _QWORD *v22; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v24; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v26; // [rsp+30h] [rbp-A0h]
  unsigned int v27; // [rsp+38h] [rbp-98h]
  unsigned __int64 v28; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v29; // [rsp+48h] [rbp-88h]
  unsigned __int64 v30; // [rsp+50h] [rbp-80h]
  unsigned int v31; // [rsp+58h] [rbp-78h]
  unsigned __int64 v32; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+68h] [rbp-68h]
  unsigned __int64 v34; // [rsp+70h] [rbp-60h]
  unsigned int v35; // [rsp+78h] [rbp-58h]
  unsigned __int64 v36; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+88h] [rbp-48h]
  unsigned __int64 v38; // [rsp+90h] [rbp-40h]
  unsigned int v39; // [rsp+98h] [rbp-38h]

  v5 = 0;
  if ( sub_33CF170(a4) )
    return v5;
  sub_33DD090((__int64)&v24, a1, a4, a5, 0);
  if ( *(_DWORD *)(a2 + 24) != 64 || (_DWORD)a3 != 1 )
    goto LABEL_4;
  v15 = v25;
  v37 = v25;
  if ( v25 <= 0x40 )
  {
    v16 = v24;
LABEL_39:
    v17 = ~v16;
    v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
    if ( !v15 )
      goto LABEL_41;
    v33 = v15;
    v32 = v18 & v17;
    if ( (v18 & v17) <= 1 )
      goto LABEL_41;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v36, (const void **)&v24);
  v15 = v37;
  if ( v37 <= 0x40 )
  {
    v16 = v36;
    goto LABEL_39;
  }
  sub_C43D10((__int64)&v36);
  v33 = v37;
  v32 = v36;
  v21 = v37;
  if ( v37 <= 0x40 )
  {
    if ( v36 <= 1 )
    {
LABEL_41:
      v5 = 0;
      goto LABEL_24;
    }
LABEL_4:
    sub_33DD090((__int64)&v28, a1, a2, a3, 0);
    if ( *(_DWORD *)(a4 + 24) != 64 || (_DWORD)a5 != 1 )
      goto LABEL_5;
    v11 = v29;
    v37 = v29;
    if ( v29 > 0x40 )
    {
      sub_C43780((__int64)&v36, (const void **)&v28);
      v11 = v37;
      if ( v37 > 0x40 )
      {
        sub_C43D10((__int64)&v36);
        v19 = v37;
        v20 = (_QWORD *)v36;
        v33 = v37;
        v32 = v36;
        if ( v37 <= 0x40 )
        {
          if ( v36 > 1 )
          {
LABEL_5:
            sub_AAF050((__int64)&v32, (__int64)&v28, 0);
            sub_AAF050((__int64)&v36, (__int64)&v24, 0);
            v10 = sub_ABD960((__int64)&v32, (__int64)&v36);
            if ( v10 > 3 )
              BUG();
            v5 = dword_44DF830[v10];
            if ( v39 > 0x40 && v38 )
              j_j___libc_free_0_0(v38);
            if ( v37 > 0x40 && v36 )
              j_j___libc_free_0_0(v36);
            if ( v35 > 0x40 && v34 )
              j_j___libc_free_0_0(v34);
            if ( v33 > 0x40 && v32 )
              j_j___libc_free_0_0(v32);
            goto LABEL_18;
          }
        }
        else
        {
          if ( v19 - (unsigned int)sub_C444A0((__int64)&v32) > 0x40 || *v20 > 1u )
          {
            if ( v32 )
              j_j___libc_free_0_0(v32);
            goto LABEL_5;
          }
          if ( v20 )
          {
            v5 = 0;
            j_j___libc_free_0_0((unsigned __int64)v20);
            goto LABEL_18;
          }
        }
LABEL_58:
        v5 = 0;
LABEL_18:
        if ( v31 > 0x40 && v30 )
          j_j___libc_free_0_0(v30);
        if ( v29 > 0x40 && v28 )
          j_j___libc_free_0_0(v28);
        goto LABEL_24;
      }
      v12 = v36;
    }
    else
    {
      v12 = v28;
    }
    v13 = ~v12;
    v14 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
    if ( v11 )
    {
      v33 = v11;
      v32 = v14 & v13;
      if ( (v14 & v13) > 1 )
        goto LABEL_5;
    }
    goto LABEL_58;
  }
  v22 = (_QWORD *)v36;
  if ( v21 - (unsigned int)sub_C444A0((__int64)&v32) > 0x40 || *v22 > 1u )
  {
    if ( v32 )
      j_j___libc_free_0_0(v32);
    goto LABEL_4;
  }
  j_j___libc_free_0_0((unsigned __int64)v22);
LABEL_24:
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  return v5;
}
