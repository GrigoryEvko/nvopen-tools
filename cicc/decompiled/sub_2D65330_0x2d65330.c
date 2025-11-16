// Function: sub_2D65330
// Address: 0x2d65330
//
__int64 __fastcall sub_2D65330(__int64 **a1, _BYTE *a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  int v4; // eax
  unsigned int v5; // r12d
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rsi
  char v14; // al
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rsi
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-70h]
  __int64 v28; // [rsp+0h] [rbp-70h]
  __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 v31; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v32; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v33; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v34; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v35; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v36; // [rsp+38h] [rbp-38h]

  if ( *a2 != 82 )
    return 0;
  v2 = *((_QWORD *)a2 - 8);
  if ( !v2 )
    return 0;
  v3 = *((_QWORD *)a2 - 4);
  if ( !v3 )
    return 0;
  v4 = sub_B53900((__int64)a2);
  if ( v4 == 36 )
  {
    if ( *(_BYTE *)v2 == 42
      && (v16 = *(_QWORD *)(v2 - 64)) != 0
      && (v17 = *(_QWORD *)(v2 - 32)) != 0
      && (v3 == v16 || v3 == v17) )
    {
      **a1 = v16;
      *a1[1] = v17;
      if ( (unsigned __int8)(*(_BYTE *)v2 - 42) > 0x11u )
        return 0;
    }
    else
    {
      v35 = 0;
      v36 = &v31;
      v11 = *(_QWORD *)(v2 + 16);
      if ( !v11 || *(_QWORD *)(v11 + 8) || *(_BYTE *)v2 != 59 )
        return 0;
      v12 = sub_995B10(&v35, *(_QWORD *)(v2 - 64));
      v13 = *(_QWORD *)(v2 - 32);
      if ( v12 && v13 )
      {
        *v36 = v13;
      }
      else
      {
        if ( !(unsigned __int8)sub_995B10(&v35, v13) )
          return 0;
        v23 = *(_QWORD *)(v2 - 64);
        if ( !v23 )
          return 0;
        *v36 = v23;
      }
      if ( !v31 )
        return 0;
      **a1 = v31;
      *a1[1] = v3;
      if ( (unsigned __int8)(*(_BYTE *)v2 - 42) > 0x11u )
        return 0;
    }
    v5 = 1;
    *a1[2] = v2;
    return v5;
  }
  if ( v4 == 34 )
  {
    if ( *(_BYTE *)v3 == 42 )
    {
      v18 = *(_QWORD *)(v3 - 64);
      if ( v18 )
      {
        v19 = *(_QWORD *)(v3 - 32);
        if ( v19 )
        {
          if ( v2 == v18 || v2 == v19 )
          {
            **a1 = v18;
            *a1[1] = v19;
            if ( (unsigned __int8)(*(_BYTE *)v3 - 42) > 0x11u )
              return 0;
LABEL_36:
            v5 = 1;
            *a1[2] = v3;
            return v5;
          }
        }
      }
    }
    v35 = 0;
    v36 = &v31;
    v9 = *(_QWORD *)(v3 + 16);
    if ( !v9 || *(_QWORD *)(v9 + 8) || *(_BYTE *)v3 != 59 )
      return 0;
    v14 = sub_995B10(&v35, *(_QWORD *)(v3 - 64));
    v15 = *(_QWORD *)(v3 - 32);
    if ( v14 && v15 )
    {
      *v36 = v15;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v35, v15) )
        return 0;
      v26 = *(_QWORD *)(v3 - 64);
      if ( !v26 )
        return 0;
      *v36 = v26;
    }
    if ( v31 )
    {
      **a1 = v31;
      *a1[1] = v2;
      if ( (unsigned __int8)(*(_BYTE *)v3 - 42) <= 0x11u )
        goto LABEL_36;
    }
    return 0;
  }
  v35 = 0;
  v36 = &v31;
  if ( v4 != 32 )
    return 0;
  if ( *(_BYTE *)v2 == 42 )
  {
    v29 = *(_QWORD *)(v2 - 64);
    if ( v29 )
    {
      v27 = *(_QWORD *)(v2 - 32);
      if ( v27 )
      {
        v32 = 0;
        v5 = sub_10081F0(&v32, v3);
        if ( (_BYTE)v5 )
        {
          v33 = 0;
          v20 = sub_993A50(&v33, v29);
          v21 = v29;
          v22 = v27;
          if ( v20 || (v24 = v27, v28 = v29, v30 = v22, v34 = 0, v25 = sub_993A50(&v34, v24), v22 = v30, v21 = v28, v25) )
          {
            **a1 = v21;
            *a1[1] = v22;
            if ( (unsigned __int8)(*(_BYTE *)v2 - 42) <= 0x11u )
            {
              *a1[2] = v2;
              return v5;
            }
            return 0;
          }
        }
      }
    }
  }
  v32 = 0;
  v5 = sub_10081F0(&v32, v2);
  if ( !(_BYTE)v5 )
    return 0;
  if ( *(_BYTE *)v3 != 42 )
    return 0;
  v6 = *(_QWORD *)(v3 - 64);
  if ( !v6 )
    return 0;
  v7 = *(_QWORD *)(v3 - 32);
  if ( !v7 )
    return 0;
  v8 = *(_QWORD *)(v3 - 64);
  v33 = 0;
  if ( !(unsigned __int8)sub_993A50(&v33, v8) )
  {
    v34 = 0;
    if ( !(unsigned __int8)sub_993A50(&v34, v7) )
      return 0;
  }
  **a1 = v6;
  *a1[1] = v7;
  if ( (unsigned __int8)(*(_BYTE *)v3 - 42) > 0x11u )
    return 0;
  *a1[2] = v3;
  return v5;
}
