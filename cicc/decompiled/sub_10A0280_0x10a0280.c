// Function: sub_10A0280
// Address: 0x10a0280
//
unsigned __int8 *__fastcall sub_10A0280(_BYTE *a1, unsigned int **a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v5; // rdx
  __int64 v6; // rdx
  _BYTE *v8; // r15
  _BYTE *v9; // r10
  _BYTE *v10; // rbx
  bool v11; // zf
  unsigned __int8 *v12; // rax
  __int64 v13; // r9
  __int64 v14; // r10
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rdi
  char v18; // al
  _BYTE *v19; // rcx
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  __int64 v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+10h] [rbp-80h]
  char v24; // [rsp+1Fh] [rbp-71h]
  __int64 v25; // [rsp+28h] [rbp-68h] BYREF
  _BYTE v26[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v27; // [rsp+50h] [rbp-40h]

  v2 = *((_QWORD *)a1 - 8);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 )
    return (unsigned __int8 *)v3;
  v5 = *(_QWORD *)(v3 + 8);
  v3 = 0;
  if ( v5 )
    return (unsigned __int8 *)v3;
  v6 = *((_QWORD *)a1 - 4);
  v3 = *(_QWORD *)(v6 + 16);
  if ( !v3 )
    return (unsigned __int8 *)v3;
  v3 = *(_QWORD *)(v3 + 8);
  if ( v3 )
    return 0;
  if ( *(_BYTE *)v2 == 47 )
  {
    v19 = *(_BYTE **)(v2 - 64);
    if ( !v19 )
      return (unsigned __int8 *)v3;
    v8 = *(_BYTE **)(v2 - 32);
    if ( !v8 )
      return (unsigned __int8 *)v3;
    if ( *(_BYTE *)v6 != 47 )
      goto LABEL_46;
    v10 = *(_BYTE **)(v6 - 64);
    v21 = *(_BYTE **)(v6 - 32);
    if ( !v10 )
      goto LABEL_46;
    if ( v21 == v8 )
    {
      v9 = v8;
      v8 = v19;
      goto LABEL_29;
    }
    if ( v21 && v8 == v10 )
    {
      v9 = v8;
      v10 = *(_BYTE **)(v6 - 32);
      v8 = v19;
    }
    else
    {
LABEL_46:
      if ( *(_BYTE *)v6 != 47 )
        return (unsigned __int8 *)v3;
      v9 = *(_BYTE **)(v6 - 64);
      v10 = *(_BYTE **)(v6 - 32);
      if ( !v9 )
        return (unsigned __int8 *)v3;
      if ( v10 == v19 )
      {
        v10 = *(_BYTE **)(v6 - 64);
        v9 = v19;
      }
      else if ( !v10 || v19 != v9 )
      {
        return (unsigned __int8 *)v3;
      }
    }
LABEL_29:
    v24 = 1;
LABEL_15:
    v11 = *a1 == 43;
    v22 = (__int64)v9;
    v27 = 257;
    if ( v11 )
    {
      sub_10A0170((__int64)&v25, (__int64)a1);
      v12 = (unsigned __int8 *)sub_92A220(a2, v8, v10, v25, (__int64)v26, 0);
    }
    else
    {
      sub_10A0170((__int64)&v25, (__int64)a1);
      v12 = (unsigned __int8 *)sub_94AB40(a2, v8, v10, v25, (__int64)v26, 0);
    }
    v14 = v22;
    v15 = (__int64)v12;
    v16 = *v12;
    v17 = (__int64)(v12 + 24);
    if ( (_BYTE)v16 != 18 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v12 + 1) + 8LL) - 17 > 1
        || (unsigned __int8)v16 > 0x15u
        || (v20 = sub_AD7630((__int64)v12, 0, v16), v14 = v22, !v20)
        || *v20 != 18 )
      {
LABEL_19:
        v27 = 257;
        if ( v24 )
          return sub_109FE60(18, v15, v14, (__int64)a1, (__int64)v26, v13, 0, 0);
        else
          return sub_109FE60(21, v15, v14, (__int64)a1, (__int64)v26, v13, 0, 0);
      }
      v17 = (__int64)(v20 + 24);
    }
    v23 = v14;
    v18 = sub_109FDD0(v17);
    v14 = v23;
    if ( !v18 )
      return (unsigned __int8 *)v3;
    goto LABEL_19;
  }
  if ( *(_BYTE *)v2 == 50 )
  {
    v8 = *(_BYTE **)(v2 - 64);
    if ( v8 )
    {
      v9 = *(_BYTE **)(v2 - 32);
      if ( v9 )
      {
        if ( *(_BYTE *)v6 == 50 )
        {
          v10 = *(_BYTE **)(v6 - 64);
          if ( v10 )
          {
            if ( v9 == *(_BYTE **)(v6 - 32) )
            {
              v24 = 0;
              goto LABEL_15;
            }
          }
        }
      }
    }
  }
  return (unsigned __int8 *)v3;
}
