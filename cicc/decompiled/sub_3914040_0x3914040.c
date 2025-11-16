// Function: sub_3914040
// Address: 0x3914040
//
__int64 __fastcall sub_3914040(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v8; // rax
  __int64 v9; // r12
  _BYTE *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  const char *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // [rsp+0h] [rbp-90h] BYREF
  __int64 v20; // [rsp+8h] [rbp-88h]
  _QWORD v21[2]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v22; // [rsp+20h] [rbp-70h]
  _QWORD v23[2]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v24; // [rsp+40h] [rbp-50h]
  __int64 v25; // [rsp+50h] [rbp-40h] BYREF
  __int64 v26; // [rsp+58h] [rbp-38h]
  __int64 v27; // [rsp+60h] [rbp-30h]
  int v28; // [rsp+68h] [rbp-28h]

  v4 = *(_QWORD *)(a2 + 24);
  *(_BYTE *)(a2 + 8) |= 4u;
  if ( *(_DWORD *)v4 != 1 )
  {
    v28 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    if ( !(unsigned __int8)sub_38CF2C0(v4, (__int64)&v25, a3, 0) )
    {
      v22 = 1283;
      v19 = sub_3913870((_BYTE *)a2);
      v15 = "unable to evaluate offset for variable '";
      v20 = v18;
LABEL_14:
      v21[0] = v15;
      v21[1] = &v19;
      v23[0] = v21;
      v23[1] = "'";
      v24 = 770;
      sub_16BCFB0((__int64)v23, 1u);
    }
    v8 = v25;
    if ( v25 )
    {
      v9 = *(_QWORD *)(v25 + 24);
      if ( (*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v11 = v26;
        if ( !v26 )
        {
          v5 = v27;
          goto LABEL_17;
        }
      }
      else
      {
        if ( (*(_BYTE *)(v9 + 9) & 0xC) != 8 )
        {
LABEL_8:
          v10 = (_BYTE *)v9;
LABEL_13:
          v13 = sub_3913870(v10);
          v22 = 1283;
          v20 = v14;
          v19 = v13;
          v15 = "unable to evaluate offset to undefined symbol '";
          goto LABEL_14;
        }
        *(_BYTE *)(v9 + 8) |= 4u;
        v17 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
        *(_QWORD *)v9 = v17 | *(_QWORD *)v9 & 7LL;
        if ( !v17 )
        {
          v9 = *(_QWORD *)(v25 + 24);
          goto LABEL_8;
        }
        v11 = v26;
        if ( !v26 )
        {
          v8 = v25;
          v5 = v27;
          if ( !v25 )
            return v5;
          goto LABEL_17;
        }
      }
    }
    else
    {
      v11 = v26;
      if ( !v26 )
        return v27;
    }
    v12 = *(_QWORD *)(v11 + 24);
    if ( (*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v8 = v25;
      v5 = v27;
      if ( !v25 )
        goto LABEL_20;
      goto LABEL_17;
    }
    if ( (*(_BYTE *)(v12 + 9) & 0xC) != 8 )
    {
LABEL_12:
      v10 = (_BYTE *)v12;
      goto LABEL_13;
    }
    *(_BYTE *)(v12 + 8) |= 4u;
    v16 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v12 + 24));
    *(_QWORD *)v12 = v16 | *(_QWORD *)v12 & 7LL;
    if ( !v16 )
    {
      v12 = *(_QWORD *)(v26 + 24);
      goto LABEL_12;
    }
    v8 = v25;
    v5 = v27;
    if ( !v25 )
    {
LABEL_18:
      if ( !v26 )
        return v5;
      v12 = *(_QWORD *)(v26 + 24);
LABEL_20:
      v5 += sub_3913FA0(a1, v12, a3);
      return v5;
    }
LABEL_17:
    v5 += sub_3913FA0(a1, *(_QWORD *)(v8 + 24), a3);
    goto LABEL_18;
  }
  return *(_QWORD *)(v4 + 16);
}
