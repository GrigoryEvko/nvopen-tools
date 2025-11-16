// Function: sub_1070D00
// Address: 0x1070d00
//
__int64 __fastcall sub_1070D00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  const char *v18; // rcx
  void *v19; // rax
  void *v20; // rax
  __int64 v21; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v22; // [rsp+8h] [rbp-C8h]
  __int64 v23; // [rsp+10h] [rbp-C0h]
  int v24; // [rsp+18h] [rbp-B8h]
  __m128i v25; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v26; // [rsp+30h] [rbp-A0h]
  __int64 v27; // [rsp+38h] [rbp-98h]
  __int16 v28; // [rsp+40h] [rbp-90h]
  __m128i v29[2]; // [rsp+50h] [rbp-80h] BYREF
  char v30; // [rsp+70h] [rbp-60h]
  char v31; // [rsp+71h] [rbp-5Fh]
  __m128i v32[5]; // [rsp+80h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 24);
  *(_BYTE *)(a2 + 8) |= 8u;
  if ( *(_BYTE *)v4 != 1 )
  {
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    if ( !(unsigned __int8)sub_E81950((int *)v4, (__int64)&v21, a3, 0) )
    {
      v31 = 1;
      v29[0].m128i_i64[0] = (__int64)"'";
      v30 = 3;
      v14 = sub_E5B9B0(a2);
      v18 = "unable to evaluate offset for variable '";
      v28 = 1283;
LABEL_19:
      v27 = v15;
      v25.m128i_i64[0] = (__int64)v18;
      v26 = v14;
      sub_9C6370(v32, &v25, v29, (__int64)v18, v16, v17);
      sub_C64D30((__int64)v32, 1u);
    }
    v9 = v21;
    if ( !v21 )
    {
      v11 = v22;
      if ( !v22 )
        return v23;
      goto LABEL_8;
    }
    v10 = *(_QWORD *)(v21 + 16);
    if ( *(_QWORD *)v10 )
    {
      v11 = v22;
      if ( v22 )
      {
LABEL_8:
        v12 = *(_QWORD *)(v11 + 16);
        if ( *(_QWORD *)v12 )
        {
          v9 = v21;
          v5 = v23;
          if ( !v21 )
            goto LABEL_13;
          goto LABEL_10;
        }
        if ( (*(_BYTE *)(v12 + 9) & 0x70) == 0x20 && *(char *)(v12 + 8) >= 0 )
        {
          *(_BYTE *)(v12 + 8) |= 8u;
          v19 = sub_E807D0(*(_QWORD *)(v12 + 24));
          *(_QWORD *)v12 = v19;
          if ( v19 )
          {
            v9 = v21;
            v5 = v23;
            if ( !v21 )
              goto LABEL_11;
            goto LABEL_10;
          }
          v11 = v22;
        }
        v31 = 1;
        v29[0].m128i_i64[0] = (__int64)"'";
        v30 = 3;
        v13 = *(_QWORD *)(v11 + 16);
        goto LABEL_18;
      }
      v5 = v23;
LABEL_10:
      v5 += sub_1070C50(a1, *(_BYTE **)(v9 + 16), a3, v8);
LABEL_11:
      if ( !v22 )
        return v5;
      v12 = *(_QWORD *)(v22 + 16);
LABEL_13:
      v5 += sub_1070C50(a1, (_BYTE *)v12, a3, v8);
      return v5;
    }
    if ( (*(_BYTE *)(v10 + 9) & 0x70) == 0x20 && *(char *)(v10 + 8) >= 0 )
    {
      *(_BYTE *)(v10 + 8) |= 8u;
      v20 = sub_E807D0(*(_QWORD *)(v10 + 24));
      *(_QWORD *)v10 = v20;
      if ( v20 )
      {
        v11 = v22;
        if ( v22 )
          goto LABEL_8;
        v9 = v21;
        v5 = v23;
        if ( !v21 )
          return v5;
        goto LABEL_10;
      }
      v9 = v21;
    }
    v31 = 1;
    v29[0].m128i_i64[0] = (__int64)"'";
    v30 = 3;
    v13 = *(_QWORD *)(v9 + 16);
LABEL_18:
    v14 = sub_E5B9B0(v13);
    v18 = "unable to evaluate offset to undefined symbol '";
    v28 = 1283;
    goto LABEL_19;
  }
  return *(_QWORD *)(v4 + 16);
}
