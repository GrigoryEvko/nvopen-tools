// Function: sub_3705770
// Address: 0x3705770
//
__int64 *__fastcall sub_3705770(__int64 *a1, __int16 *a2, __int64 *a3)
{
  unsigned __int64 v4; // rax
  __int16 v5; // ax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int64 v21; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int64 v22; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h]
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h]

  (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *))(*a3 + 56))(&v22, a3, a2);
  v4 = v22 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_11;
  v22 = 0;
  sub_9C66B0((__int64 *)&v22);
  v5 = *a2;
  if ( (unsigned __int16)*a2 <= 0x1409u )
  {
    if ( (unsigned __int16)v5 > 0x13FFu )
    {
      switch ( v5 )
      {
        case 5120:
LABEL_10:
          sub_37056D0(&v22, a2, (__int64)a3);
          v4 = v22 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_11;
          goto LABEL_6;
        case 5121:
        case 5122:
          sub_3704CB0(&v22, a2, a3);
          v4 = v22 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_11;
          goto LABEL_6;
        case 5124:
          LOWORD(v20) = 5124;
          v9 = *a3;
          *(_DWORD *)((char *)&v20 + 2) = 0;
          (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v9 + 264))(
            &v22,
            a3,
            a2,
            &v20);
          v8 = v22 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_19;
          goto LABEL_16;
        case 5129:
          v7 = *a3;
          LOWORD(v20) = 5129;
          *(_DWORD *)((char *)&v20 + 2) = 0;
          (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v7 + 208))(
            &v22,
            a3,
            a2,
            &v20);
          v8 = v22 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
LABEL_19:
            v21 = 0;
            v22 = v8 | 1;
            sub_9C6670((__int64 *)&v21, &v22);
            sub_9C66B0((__int64 *)&v22);
            v10 = v21 & 0xFFFFFFFFFFFFFFFELL;
            if ( (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
              *a1 = 0;
              v21 = v10 | 1;
              sub_9C6670(a1, &v21);
              sub_9C66B0((__int64 *)&v21);
              return a1;
            }
          }
          else
          {
LABEL_16:
            v22 = 0;
            sub_9C66B0((__int64 *)&v22);
            v21 = 1;
            sub_9C66B0((__int64 *)&v22);
          }
          v21 = 0;
          sub_9C66B0((__int64 *)&v21);
          goto LABEL_7;
        default:
          break;
      }
    }
LABEL_5:
    (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *))(*a3 + 48))(&v22, a3, a2);
    v4 = v22 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v22 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
LABEL_6:
      v22 = 0;
      sub_9C66B0((__int64 *)&v22);
LABEL_7:
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *))(*a3 + 64))(&v22, a3, a2);
      v4 = v22 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v22 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      {
        v22 = 0;
        sub_9C66B0((__int64 *)&v22);
        *a1 = 1;
        sub_9C66B0((__int64 *)&v22);
        return a1;
      }
    }
LABEL_11:
    *a1 = 0;
    v22 = v4 | 1;
    sub_9C6670(a1, &v22);
    sub_9C66B0((__int64 *)&v22);
    return a1;
  }
  switch ( v5 )
  {
    case 5378:
      v15 = *a3;
      v22 = 5378;
      v23 = 0;
      v24 = 1;
      v25 = 0;
      v26 = 0;
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v15 + 256))(
        &v21,
        a3,
        a2,
        &v22);
      v16 = v21 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v21 = 0;
        v20 = v16 | 1;
      }
      else
      {
        v20 = 1;
        v21 = 0;
      }
      sub_9C66B0((__int64 *)&v21);
      if ( (unsigned int)v24 > 0x40 && v23 )
        j_j___libc_free_0_0(v23);
      v12 = v20 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_25;
      goto LABEL_23;
    case 5389:
      v14 = *a3;
      v22 = 5389;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v14 + 232))(
        &v21,
        a3,
        a2,
        &v22);
      v12 = v21 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_25;
      goto LABEL_22;
    case 5390:
      v17 = *a3;
      v22 = 5390;
      v23 = 0;
      v24 = 0;
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v17 + 216))(
        &v21,
        a3,
        a2,
        &v22);
      v12 = v21 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_25;
      v21 = 0;
      sub_9C66B0((__int64 *)&v21);
      v20 = 0;
      sub_9C66B0((__int64 *)&v20);
      goto LABEL_7;
    case 5391:
      v18 = *a3;
      v22 = 5391;
      v23 = 0;
      v24 = 0;
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v18 + 224))(
        &v21,
        a3,
        a2,
        &v22);
      v19 = v21 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v21 & 0xFFFFFFFFFFFFFFFELL) == 0 )
        goto LABEL_23;
      v21 = 0;
      v20 = v19 | 1;
      sub_9C66B0((__int64 *)&v21);
      goto LABEL_26;
    case 5392:
      v13 = *a3;
      v22 = 5392;
      v23 = 0;
      v24 = 0;
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v13 + 240))(
        &v21,
        a3,
        a2,
        &v22);
      v12 = v21 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_25;
      goto LABEL_22;
    case 5393:
      v11 = *a3;
      v22 = 5393;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, __int16 *, unsigned __int64 *))(v11 + 248))(
        &v21,
        a3,
        a2,
        &v22);
      v12 = v21 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v21 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      {
LABEL_22:
        v20 = 1;
        v21 = 0;
        sub_9C66B0((__int64 *)&v21);
LABEL_23:
        v20 = 0;
        sub_9C66B0((__int64 *)&v20);
        goto LABEL_7;
      }
LABEL_25:
      v20 = v12 | 1;
LABEL_26:
      *a1 = 0;
      sub_9C6670(a1, &v20);
      sub_9C66B0((__int64 *)&v20);
      break;
    case 5402:
      goto LABEL_10;
    default:
      goto LABEL_5;
  }
  return a1;
}
