// Function: sub_8AF210
// Address: 0x8af210
//
__int64 __fastcall sub_8AF210(__m128i *a1, _QWORD *a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
{
  char v7; // cl
  __int64 i; // r14
  __int64 *v9; // rsi
  __int64 **v10; // rdx
  int v11; // r12d
  __int64 **v12; // rcx
  __int64 *v13; // rax
  __int64 *v14; // r15
  int v15; // r13d
  _QWORD *v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v22; // r10d
  __int64 v26; // [rsp+38h] [rbp-C8h]
  bool v27; // [rsp+43h] [rbp-BDh]
  _BOOL4 v28; // [rsp+44h] [rbp-BCh]
  int v29; // [rsp+48h] [rbp-B8h]
  bool v30; // [rsp+4Fh] [rbp-B1h]
  unsigned int v31; // [rsp+5Ch] [rbp-A4h] BYREF
  __int64 **v32; // [rsp+60h] [rbp-A0h] BYREF
  __int64 **v33; // [rsp+68h] [rbp-98h] BYREF
  __m128i v34[9]; // [rsp+70h] [rbp-90h] BYREF

  v7 = *(_BYTE *)(a4 + 80);
  v29 = a3 & 8;
  v27 = v29 == 0 && unk_4D04310 != 0;
  if ( v27 )
  {
    if ( v7 != 20 )
    {
      i = 0;
      v27 = 0;
      v28 = 0;
      v30 = (a3 & 8) == 0 && unk_4D04310 != 0;
      goto LABEL_7;
    }
  }
  else
  {
    v30 = 0;
    if ( v7 != 20 )
      goto LABEL_65;
  }
  v30 = (a3 & 8) == 0 && unk_4D04310 != 0;
  v27 = (*(_BYTE *)(a5 + 424) & 0x10) != 0;
  if ( (a3 & 8) == 0 )
  {
LABEL_65:
    v28 = 0;
    i = 0;
    goto LABEL_7;
  }
  for ( i = *(_QWORD *)(*(_QWORD *)(a5 + 176) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v28 = (unsigned __int8)sub_877F80(a4) == 3;
LABEL_7:
  sub_89ED70((__int64)a2, (__int64)a1, &v32, &v33);
  v9 = (__int64 *)v33;
  if ( v33 )
  {
    v26 = i;
LABEL_12:
    v10 = v32;
    v11 = *((_DWORD *)v32 + 15);
    if ( *((_BYTE *)v9 + 8) == 3 )
    {
      v9 = (__int64 *)*v9;
      if ( v9 && (v9[3] & 8) != 0 )
      {
        v33 = (__int64 **)v9;
        goto LABEL_22;
      }
      goto LABEL_11;
    }
    if ( !v9 )
      return ((_BYTE)v32[7] & 0x10) != 0;
    while ( 1 )
    {
LABEL_22:
      if ( *((_BYTE *)v9 + 8) == 1 && (v9[3] & 1) != 0 )
      {
        if ( !v10 )
          return 0;
        goto LABEL_16;
      }
      if ( !v10 )
        return 0;
      if ( !v30 )
        break;
      if ( v9[4] )
        goto LABEL_16;
      sub_8AEEA0(a4, (__int64)v9, (__int64)v10, a2);
      v12 = v33;
      v10 = v32;
      if ( !v33 )
        goto LABEL_26;
      v10 = v32;
      if ( (*((_BYTE *)v33 + 8) != 1 || ((_BYTE)v33[3] & 1) == 0) && !v33[4] )
        goto LABEL_26;
      if ( ((_BYTE)v32[7] & 2) != 0 )
      {
        v31 = 0;
        sub_892150(v34);
        sub_8A4F30(
          (__int64)v33,
          (__int64)v32,
          a1,
          (__int64)a2,
          a1,
          (__int64)a2,
          (__int64 *)(a4 + 48),
          a3,
          (int *)&v31,
          v34);
        if ( v31 || !v27 && (unsigned int)sub_88D7A0((__int64)v33, v31, v17, v18, v19, v20) )
        {
          v10 = v32;
          v22 = 0;
          goto LABEL_31;
        }
        goto LABEL_16;
      }
LABEL_17:
      if ( ((_BYTE)v12[3] & 8) == 0 )
        goto LABEL_11;
      v9 = *v12;
      v13 = *v32;
      if ( !*v32 || (v13[7] & 0x40) == 0 || v11 != *((_DWORD *)v13 + 15) )
      {
        if ( v9 && (v9[3] & 8) != 0 && ((_BYTE)v32[7] & 0x10) == 0 )
          return 0;
LABEL_11:
        sub_89ED80(&v32, &v33);
        v9 = (__int64 *)v33;
        if ( !v33 )
          goto LABEL_66;
        goto LABEL_12;
      }
      v33 = (__int64 **)*v12;
      v10 = (__int64 **)*v32;
      v32 = (__int64 **)*v32;
      if ( !v9 )
      {
        v22 = 1;
        goto LABEL_31;
      }
    }
    if ( !v9[4] )
    {
LABEL_26:
      if ( !v29 )
        goto LABEL_30;
      v14 = v10[1];
      if ( v28 )
      {
        if ( (unsigned int)sub_88FAD0(*((_BYTE *)v14 + 80), v14[11], *(_QWORD *)(v26 + 160), 0) )
          goto LABEL_29;
      }
      else
      {
        v15 = a6;
        v16 = **(_QWORD ***)(v26 + 168);
        if ( v16 )
        {
          while ( 1 )
          {
            if ( a6 )
            {
              if ( !v15 )
                goto LABEL_16;
              --v15;
            }
            if ( (unsigned int)sub_88FAD0(*((_BYTE *)v14 + 80), v14[11], v16[1], 0) )
              break;
            v16 = (_QWORD *)*v16;
            if ( !v16 )
              goto LABEL_16;
          }
LABEL_29:
          v10 = v32;
LABEL_30:
          v22 = 0;
          goto LABEL_31;
        }
      }
    }
LABEL_16:
    v12 = v33;
    goto LABEL_17;
  }
LABEL_66:
  v10 = v32;
  v22 = 1;
LABEL_31:
  if ( v10 && ((_BYTE)v10[7] & 0x10) == 0 )
    return 0;
  return v22;
}
