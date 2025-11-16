// Function: sub_2FCCAF0
// Address: 0x2fccaf0
//
__int64 __fastcall sub_2FCCAF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // edx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  unsigned __int64 *v15; // [rsp+0h] [rbp-F0h]
  unsigned int v16; // [rsp+8h] [rbp-E8h]
  __int64 v17; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v19; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v21; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v23; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v24; // [rsp+58h] [rbp-98h]
  __int64 v25; // [rsp+60h] [rbp-90h]
  char v26; // [rsp+70h] [rbp-80h]
  __m128i v27; // [rsp+80h] [rbp-70h] BYREF
  char v28; // [rsp+B0h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 16);
  v19 = a2;
  v20 = a3;
  v17 = a4 + 312;
  if ( !v5 )
    return 0;
  v7 = v5;
  while ( 2 )
  {
    v8 = *(_QWORD *)(v7 + 24);
    sub_D66840(&v27, (_BYTE *)v8);
    if ( v28
      && v27.m128i_i64[1] != -1
      && v27.m128i_i64[1] != 0xBFFFFFFFFFFFFFFELL
      && ((_BYTE)v20 != 1 && (v27.m128i_i64[1] & 0x4000000000000000LL) != 0
       || v19 < (v27.m128i_i64[1] & 0x3FFFFFFFFFFFFFFFuLL)) )
    {
      return 1;
    }
    switch ( *(_BYTE *)v8 )
    {
      case 0x1E:
      case 0x3D:
        goto LABEL_14;
      case 0x3E:
        v10 = *(_QWORD *)(v8 - 64);
        if ( a1 != v10 )
          goto LABEL_14;
        goto LABEL_13;
      case 0x3F:
        v22 = sub_AE43F0(v17, *(_QWORD *)(v8 + 8));
        if ( v22 > 0x40 )
          sub_C43690((__int64)&v21, 0, 0);
        else
          v21 = 0;
        if ( !(unsigned __int8)sub_B4DE60(v8, v17, (__int64)&v21) )
          goto LABEL_10;
        v16 = v22;
        if ( v22 > 0x40 )
        {
          v15 = (unsigned __int64 *)v21;
          if ( v16 - (unsigned int)sub_C444A0((__int64)&v21) > 0x40 )
            goto LABEL_43;
          v14 = *v15;
        }
        else
        {
          v14 = v21;
        }
        if ( v19 > v14 )
        {
          LOBYTE(v24) = 0;
          v23 = v19 - v14;
          if ( !(unsigned __int8)sub_2FCCAF0(v8, v19 - v14, v24, a4, a5) )
          {
            if ( v22 > 0x40 )
            {
              if ( v21 )
                j_j___libc_free_0_0(v21);
            }
LABEL_14:
            v7 = *(_QWORD *)(v7 + 8);
            if ( !v7 )
              return 0;
            continue;
          }
        }
LABEL_10:
        if ( v22 > 0x40 )
        {
LABEL_43:
          if ( v21 )
            j_j___libc_free_0_0(v21);
        }
        return 1;
      case 0x41:
      case 0x42:
      case 0x4C:
        v10 = *(_QWORD *)(v8 - 32);
        if ( a1 != v10 )
          goto LABEL_14;
LABEL_13:
        if ( !v10 )
          goto LABEL_14;
        return 1;
      case 0x4E:
      case 0x4F:
      case 0x56:
        if ( (unsigned __int8)sub_2FCCAF0(v8, v19, v20, a4, a5) )
          return 1;
        goto LABEL_14;
      case 0x54:
        v21 = v8;
        sub_2FCC700((__int64)&v23, a5, (__int64 *)&v21, (__int64 *)&v19);
        if ( v26 )
          goto LABEL_23;
        v11 = v25;
        if ( !(_BYTE)v20 && *(_BYTE *)(v25 + 16) || v19 < *(_QWORD *)(v25 + 8) )
        {
          v12 = *(_DWORD *)(v25 + 24);
          if ( v12 == 3 )
            return 1;
          v13 = v19;
          *(_DWORD *)(v25 + 24) = v12 + 1;
          *(_QWORD *)(v11 + 8) = v13;
          *(_BYTE *)(v11 + 16) = v20;
LABEL_23:
          if ( (unsigned __int8)sub_2FCCAF0(v21, v19, v20, a4, a5) )
            return 1;
        }
        goto LABEL_14;
      case 0x55:
        if ( !sub_B46AA0(v8) && !sub_B46A10(v8) )
          return 1;
        goto LABEL_14;
      default:
        return 1;
    }
  }
}
