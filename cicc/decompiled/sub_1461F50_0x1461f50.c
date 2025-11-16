// Function: sub_1461F50
// Address: 0x1461f50
//
_BYTE *__fastcall sub_1461F50(_QWORD *a1, __int64 a2, __int64 *a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r15
  _BYTE *v12; // r12
  __int64 v14; // rbx
  _BYTE *v15; // r15
  __int64 v16; // rdx
  _BYTE **v17; // rcx
  _BYTE *v18; // rdi
  _BYTE *v19; // rax
  _BYTE *v20; // rdx
  __int64 v21; // rcx
  _BYTE **v22; // rsi
  _BYTE *v23; // rdi
  _BYTE *v24; // rax
  _QWORD *v25; // [rsp+8h] [rbp-68h]
  _QWORD *v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  _BYTE *v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+20h] [rbp-50h] BYREF
  __int64 v31; // [rsp+28h] [rbp-48h]
  __int64 v32; // [rsp+30h] [rbp-40h]

  v32 = a2;
  v30 = (__int64)&v30;
  v31 = 1;
  v4 = sub_1461E10(a1, (__int64)&v30);
  v30 = (__int64)&v30;
  v5 = v4;
  v6 = *a3;
  v31 = 1;
  v7 = a1 + 1;
  v32 = v6;
  v8 = sub_1461E10(a1, (__int64)&v30);
  if ( a1 + 1 == v8 )
  {
    v12 = 0;
    v10 = 0;
    if ( v7 == v5 )
      return v12;
    goto LABEL_11;
  }
  v9 = v8;
  v10 = (unsigned __int64)(v8 + 4);
  if ( (v8[5] & 1) != 0 || (v10 = v8[4], (*(_BYTE *)(v10 + 8) & 1) != 0) )
  {
    if ( v7 == v5 )
    {
      v12 = 0;
      goto LABEL_8;
    }
    goto LABEL_11;
  }
  v11 = *(_QWORD *)v10;
  if ( (*(_BYTE *)(*(_QWORD *)v10 + 8LL) & 1) != 0 )
  {
    v10 = *(_QWORD *)v10;
  }
  else
  {
    v20 = *(_BYTE **)v11;
    if ( (*(_BYTE *)(*(_QWORD *)v11 + 8LL) & 1) == 0 )
    {
      v21 = *(_QWORD *)v20;
      if ( (*(_BYTE *)(*(_QWORD *)v20 + 8LL) & 1) != 0 )
      {
        v20 = *(_BYTE **)v20;
      }
      else
      {
        v22 = *(_BYTE ***)v21;
        if ( (*(_BYTE *)(*(_QWORD *)v21 + 8LL) & 1) == 0 )
        {
          v23 = *v22;
          v25 = *(_QWORD **)v21;
          v22 = (_BYTE **)v23;
          if ( (v23[8] & 1) == 0 )
          {
            v27 = *(_QWORD *)v20;
            v29 = *(_BYTE **)v11;
            v24 = sub_145F440(v23);
            v20 = v29;
            v21 = v27;
            *v25 = v24;
            v22 = (_BYTE **)v24;
          }
          *(_QWORD *)v21 = v22;
        }
        *(_QWORD *)v20 = v22;
        v20 = v22;
      }
      *(_QWORD *)v11 = v20;
    }
    *(_QWORD *)v10 = v20;
    v10 = (unsigned __int64)v20;
  }
  v9[4] = v10;
  v12 = 0;
  if ( v7 != v5 )
  {
LABEL_11:
    v12 = v5 + 4;
    if ( (v5[5] & 1) == 0 )
    {
      v12 = (_BYTE *)v5[4];
      if ( (v12[8] & 1) == 0 )
      {
        v14 = *(_QWORD *)v12;
        if ( (*(_BYTE *)(*(_QWORD *)v12 + 8LL) & 1) != 0 )
        {
          v12 = *(_BYTE **)v12;
        }
        else
        {
          v15 = *(_BYTE **)v14;
          if ( (*(_BYTE *)(*(_QWORD *)v14 + 8LL) & 1) == 0 )
          {
            v16 = *(_QWORD *)v15;
            if ( (*(_BYTE *)(*(_QWORD *)v15 + 8LL) & 1) != 0 )
            {
              v15 = *(_BYTE **)v15;
            }
            else
            {
              v17 = *(_BYTE ***)v16;
              if ( (*(_BYTE *)(*(_QWORD *)v16 + 8LL) & 1) == 0 )
              {
                v18 = *v17;
                v26 = *(_QWORD **)v16;
                v17 = (_BYTE **)v18;
                if ( (v18[8] & 1) == 0 )
                {
                  v28 = *(_QWORD *)v15;
                  v19 = sub_145F440(v18);
                  v16 = v28;
                  *v26 = v19;
                  v17 = (_BYTE **)v19;
                }
                *(_QWORD *)v16 = v17;
              }
              *(_QWORD *)v15 = v17;
              v15 = v17;
            }
            *(_QWORD *)v14 = v15;
          }
          *(_QWORD *)v12 = v15;
          v12 = v15;
        }
        v5[4] = v12;
      }
    }
  }
  if ( (_BYTE *)v10 != v12 )
  {
LABEL_8:
    *(_QWORD *)(*(_QWORD *)v12 + 8LL) = v10 | *(_QWORD *)(*(_QWORD *)v12 + 8LL) & 1LL;
    *(_QWORD *)v12 = *(_QWORD *)v10;
    *(_QWORD *)(v10 + 8) &= ~1uLL;
    *(_QWORD *)v10 = v12;
  }
  return v12;
}
