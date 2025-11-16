// Function: sub_109F890
// Address: 0x109f890
//
_BYTE *__fastcall sub_109F890(__int64 *a1, __int64 a2)
{
  unsigned int v3; // eax
  int v4; // ecx
  unsigned int v5; // r15d
  _BYTE *v6; // r13
  int v8; // eax
  int v9; // r10d
  _QWORD *v10; // r13
  __int64 v11; // rax
  _BYTE *v12; // rax
  __int64 **v13; // rdi
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // [rsp+10h] [rbp-1A0h]
  int v19; // [rsp+20h] [rbp-190h]
  int v20; // [rsp+28h] [rbp-188h]
  _BYTE *v21; // [rsp+30h] [rbp-180h] BYREF
  int v22; // [rsp+38h] [rbp-178h]
  __int64 v23[4]; // [rsp+40h] [rbp-170h] BYREF
  __int64 v24; // [rsp+60h] [rbp-150h] BYREF
  int v25; // [rsp+68h] [rbp-148h]
  __int64 v26[4]; // [rsp+70h] [rbp-140h] BYREF
  __int64 v27; // [rsp+90h] [rbp-120h] BYREF
  int v28; // [rsp+98h] [rbp-118h]
  __int64 v29[4]; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v30; // [rsp+C0h] [rbp-F0h] BYREF
  int v31; // [rsp+C8h] [rbp-E8h]
  __int64 v32[4]; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v33; // [rsp+F0h] [rbp-C0h] BYREF
  int v34; // [rsp+F8h] [rbp-B8h]
  __int64 v35[4]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+120h] [rbp-90h] BYREF
  int v37; // [rsp+128h] [rbp-88h]
  __int64 v38[4]; // [rsp+130h] [rbp-80h] BYREF
  __int64 **v39; // [rsp+150h] [rbp-60h] BYREF
  __int64 v40; // [rsp+158h] [rbp-58h]
  __int64 *v41; // [rsp+160h] [rbp-50h] BYREF
  __int64 *v42; // [rsp+168h] [rbp-48h]
  __int64 *v43; // [rsp+170h] [rbp-40h]

  a1[1] = a2;
  v21 = 0;
  v22 = 0;
  v24 = 0;
  v25 = 0;
  v27 = 0;
  v28 = 0;
  v30 = 0;
  v31 = 0;
  v33 = 0;
  v34 = 0;
  v36 = 0;
  v37 = 0;
  v3 = sub_109F4E0((char *)a2, (__int64)&v21, (__int64)&v24);
  v4 = 0;
  v5 = v3;
  if ( v21 )
    v4 = sub_109F820((__int64)&v21, (__int64)&v27, (__int64)&v30);
  if ( v5 != 2 )
  {
    if ( !(_BYTE)v22 )
    {
      v6 = v21;
      if ( HIWORD(v22) == 1 )
      {
LABEL_6:
        if ( !BYTE1(v37) )
          goto LABEL_7;
        goto LABEL_30;
      }
    }
LABEL_5:
    v6 = 0;
    goto LABEL_6;
  }
  if ( v24 )
  {
    v19 = v4;
    v8 = sub_109F820((__int64)&v24, (__int64)&v33, (__int64)&v36);
    v4 = v19;
    v9 = v8;
    if ( v19 )
    {
      if ( v8 )
      {
        v41 = &v27;
        v39 = &v41;
        v42 = &v33;
        v40 = 0x400000002LL;
        if ( v19 == 2 )
        {
          LODWORD(v40) = 3;
          v43 = &v30;
        }
        if ( v8 == 2 )
        {
          (&v41)[(unsigned int)v40] = &v36;
          LODWORD(v40) = v40 + 1;
        }
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v10 = *(_QWORD **)(a2 - 8);
        else
          v10 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        if ( *(_BYTE *)*v10 > 0x15u
          && (v11 = *(_QWORD *)(*v10 + 16LL)) != 0
          && !*(_QWORD *)(v11 + 8)
          && (v16 = v10[4], *(_BYTE *)v16 > 0x15u)
          && (v17 = *(_QWORD *)(v16 + 16)) != 0 )
        {
          if ( *(_QWORD *)(v17 + 8) )
            v5 = 1;
        }
        else
        {
          v5 = 1;
        }
        v18 = v9;
        v12 = sub_109E3B0(a1, (__int64)&v39, v5);
        v13 = v39;
        v4 = v19;
        v6 = v12;
        v9 = v18;
        if ( v12 )
          goto LABEL_28;
        if ( v39 != &v41 )
        {
          _libc_free(v39, &v39);
          v4 = v19;
          v9 = v18;
        }
        goto LABEL_38;
      }
    }
    else if ( v8 )
    {
LABEL_38:
      v39 = &v41;
      v41 = (__int64 *)&v21;
      v42 = &v33;
      v40 = 0x400000002LL;
      if ( v9 == 2 )
      {
        LODWORD(v40) = 3;
        v43 = &v36;
      }
      v20 = v4;
      v14 = sub_109E3B0(a1, (__int64)&v39, 1u);
      v13 = v39;
      v4 = v20;
      v6 = v14;
      if ( v14 )
        goto LABEL_28;
      if ( v39 != &v41 )
      {
        _libc_free(v39, &v39);
        v4 = v20;
      }
    }
  }
  if ( !v4 )
    goto LABEL_5;
  v41 = &v24;
  v42 = &v27;
  v39 = &v41;
  v40 = 0x400000002LL;
  if ( v4 == 2 )
  {
    LODWORD(v40) = 3;
    v43 = &v30;
  }
  v15 = sub_109E3B0(a1, (__int64)&v39, 1u);
  v13 = v39;
  v6 = v15;
LABEL_28:
  if ( v13 == &v41 )
    goto LABEL_6;
  _libc_free(v13, &v39);
  if ( !BYTE1(v37) )
  {
LABEL_7:
    if ( !BYTE1(v34) )
      goto LABEL_8;
    goto LABEL_31;
  }
LABEL_30:
  sub_91D830(v38);
  if ( !BYTE1(v34) )
  {
LABEL_8:
    if ( !BYTE1(v31) )
      goto LABEL_9;
    goto LABEL_32;
  }
LABEL_31:
  sub_91D830(v35);
  if ( !BYTE1(v31) )
  {
LABEL_9:
    if ( !BYTE1(v28) )
      goto LABEL_10;
    goto LABEL_33;
  }
LABEL_32:
  sub_91D830(v32);
  if ( !BYTE1(v28) )
  {
LABEL_10:
    if ( !BYTE1(v25) )
      goto LABEL_11;
LABEL_34:
    sub_91D830(v26);
    if ( !BYTE1(v22) )
      return v6;
LABEL_35:
    sub_91D830(v23);
    return v6;
  }
LABEL_33:
  sub_91D830(v29);
  if ( BYTE1(v25) )
    goto LABEL_34;
LABEL_11:
  if ( BYTE1(v22) )
    goto LABEL_35;
  return v6;
}
