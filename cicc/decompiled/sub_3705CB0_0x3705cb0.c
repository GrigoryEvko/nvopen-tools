// Function: sub_3705CB0
// Address: 0x3705cb0
//
__int64 *__fastcall sub_3705CB0(__int64 *a1, __int64 **a2, _QWORD *a3)
{
  __int64 *v6; // rsi
  unsigned __int16 v7; // ax
  unsigned __int64 v8; // rax
  unsigned __int64 *v9; // r15
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-78h] BYREF
  unsigned __int64 v19; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 *v20; // [rsp+28h] [rbp-68h]
  __int64 v21; // [rsp+30h] [rbp-60h]
  unsigned __int64 v22; // [rsp+38h] [rbp-58h] BYREF
  __int64 v23; // [rsp+40h] [rbp-50h]
  __int64 v24; // [rsp+48h] [rbp-48h]
  __int64 v25; // [rsp+50h] [rbp-40h]

  v6 = *a2;
  if ( a3[1] <= 3u )
    goto LABEL_31;
  v7 = *(_WORD *)(*a3 + 2LL);
  if ( v7 > 0x151Du )
  {
    switch ( v7 )
    {
      case 0x1601u:
        v19 = 5633;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 272))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        goto LABEL_49;
      case 0x1602u:
        v19 = 5634;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 280))(
          &v18,
          v6,
          a3,
          &v19);
        v16 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_66;
        goto LABEL_47;
      case 0x1603u:
        LOWORD(v19) = 5635;
        v20 = &v22;
        v21 = 0x500000000LL;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 288))(
          &v18,
          v6,
          a3,
          &v19);
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v17 = v18 & 0xFFFFFFFFFFFFFFFELL | 1;
        }
        else
        {
          v18 = 0;
          sub_9C66B0((__int64 *)&v18);
          v17 = 1;
        }
        if ( v20 != &v22 )
          _libc_free((unsigned __int64)v20);
        goto LABEL_42;
      case 0x1604u:
        v19 = 5636;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 296))(
          &v18,
          v6,
          a3,
          &v19);
        v10 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_85;
        v18 = 0;
        sub_9C66B0((__int64 *)&v18);
        v17 = 1;
        break;
      case 0x1605u:
        v19 = 5637;
        v20 = 0;
        v21 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 304))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        goto LABEL_17;
      case 0x1606u:
        v19 = 5638;
        v20 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 312))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        goto LABEL_17;
      case 0x1607u:
        v19 = 5639;
        v20 = 0;
        LOWORD(v21) = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 320))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        goto LABEL_17;
      default:
        goto LABEL_31;
    }
LABEL_57:
    v14 = (unsigned __int64)v20;
    if ( v20 )
LABEL_41:
      j_j___libc_free_0(v14);
LABEL_42:
    v8 = v17 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_43;
    goto LABEL_47;
  }
  if ( v7 > 0x1502u )
  {
    switch ( v7 )
    {
      case 0x1503u:
        v19 = 5379;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        v23 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 128))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        goto LABEL_49;
      case 0x1504u:
      case 0x1505u:
      case 0x1519u:
        sub_3704D50(&v19, a3, v6);
        v11 = v19 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_36;
        goto LABEL_32;
      case 0x1506u:
        v19 = 5382;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        v23 = 0;
        v24 = 0;
        v25 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 144))(
          &v18,
          v6,
          a3,
          &v19);
        v16 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_66;
        goto LABEL_17;
      case 0x1507u:
        v19 = 5383;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        v23 = 0;
        v24 = 0;
        LODWORD(v25) = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 152))(
          &v18,
          v6,
          a3,
          &v19);
        v16 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_66;
        goto LABEL_47;
      case 0x1509u:
        v19 = 5385;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 336))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        goto LABEL_49;
      case 0x1515u:
        v19 = 5397;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        v23 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 160))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        goto LABEL_49;
      case 0x151Du:
        v19 = 5405;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        v23 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 168))(
          &v18,
          v6,
          a3,
          &v19);
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v17 = v18 & 0xFFFFFFFFFFFFFFFELL | 1;
        }
        else
        {
          v18 = 0;
          sub_9C66B0((__int64 *)&v18);
          v17 = 1;
          sub_9C66B0((__int64 *)&v18);
        }
        v14 = v21;
        if ( v21 )
          goto LABEL_41;
        goto LABEL_42;
      default:
        goto LABEL_31;
    }
  }
  if ( v7 == 4104 )
  {
    v19 = 4104;
    WORD2(v20) = 0;
    LODWORD(v20) = 0;
    (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 88))(&v18, v6, a3, &v19);
    v16 = v18 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_66:
      v18 = 0;
      v17 = v16 | 1;
      sub_9C66B0((__int64 *)&v18);
      goto LABEL_44;
    }
    goto LABEL_47;
  }
  if ( v7 > 0x1008u )
  {
    if ( v7 != 4611 )
    {
      if ( v7 > 0x1203u )
      {
        if ( v7 != 4613 )
        {
          if ( v7 == 4614 )
          {
            v19 = 4614;
            v20 = 0;
            v21 = 0;
            v22 = 0;
            (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 328))(
              &v18,
              v6,
              a3,
              &v19);
            if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
              v17 = v18 & 0xFFFFFFFFFFFFFFFELL | 1;
            }
            else
            {
              v18 = 0;
              sub_9C66B0((__int64 *)&v18);
              v17 = 1;
              sub_9C66B0((__int64 *)&v18);
            }
            goto LABEL_57;
          }
          goto LABEL_31;
        }
        v19 = 4613;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 184))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
LABEL_49:
        v17 = 1;
        v18 = 0;
LABEL_50:
        sub_9C66B0((__int64 *)&v18);
        v17 = 0;
        sub_9C66B0((__int64 *)&v17);
        goto LABEL_33;
      }
      if ( v7 == 4105 )
      {
        v19 = 4105;
        v20 = 0;
        v21 = 0;
        LODWORD(v22) = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 96))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_43;
        v18 = 0;
        sub_9C66B0((__int64 *)&v18);
        v17 = 1;
        goto LABEL_50;
      }
      if ( v7 == 4609 )
      {
        v19 = 4609;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 112))(
          &v18,
          v6,
          a3,
          &v19);
        v10 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
LABEL_85:
          v18 = 0;
          v17 = v10 | 1;
          sub_9C66B0((__int64 *)&v18);
        }
        else
        {
          v17 = 1;
          v18 = 0;
          sub_9C66B0((__int64 *)&v18);
        }
        goto LABEL_57;
      }
LABEL_31:
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *))(*v6 + 16))(&v19, v6, a3);
      v11 = v19 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_36;
LABEL_32:
      v19 = 0;
      sub_9C66B0((__int64 *)&v19);
LABEL_33:
      (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *))(**a2 + 40))(&v19, *a2, a3);
      v11 = v19 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v19 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      {
        v19 = 0;
        sub_9C66B0((__int64 *)&v19);
        *a1 = 1;
        sub_9C66B0((__int64 *)&v19);
        return a1;
      }
LABEL_36:
      *a1 = 0;
      v19 = v11 | 1;
      sub_9C6670(a1, &v19);
      sub_9C66B0((__int64 *)&v19);
      return a1;
    }
    v19 = 4611;
    v20 = 0;
    v21 = 0;
    (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 120))(
      &v18,
      v6,
      a3,
      &v19);
    v15 = v18 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_90:
      v9 = &v17;
      v17 = 0;
      v18 = v15 | 1;
      sub_9C6670((__int64 *)&v17, &v18);
      sub_9C66B0((__int64 *)&v18);
      if ( (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v17 = v17 & 0xFFFFFFFFFFFFFFFELL | 1;
        goto LABEL_26;
      }
LABEL_47:
      v17 = 0;
      sub_9C66B0((__int64 *)&v17);
      goto LABEL_33;
    }
LABEL_46:
    v18 = 0;
    sub_9C66B0((__int64 *)&v18);
    v17 = 1;
    sub_9C66B0((__int64 *)&v18);
    goto LABEL_47;
  }
  if ( v7 == 20 )
  {
    v19 = 20;
    (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 344))(
      &v18,
      v6,
      a3,
      &v19);
    v15 = v18 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_90;
    goto LABEL_46;
  }
  if ( v7 > 0x14u )
  {
    if ( v7 != 4097 )
    {
      if ( v7 == 4098 )
      {
        v19 = 4098;
        v20 = 0;
        LODWORD(v21) = 0;
        (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 72))(
          &v18,
          v6,
          a3,
          &v19);
        v8 = v18 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
LABEL_43:
          v17 = v8 | 1;
LABEL_44:
          *a1 = 0;
          sub_9C6670(a1, &v17);
          sub_9C66B0((__int64 *)&v17);
          return a1;
        }
LABEL_17:
        v18 = 0;
        sub_9C66B0((__int64 *)&v18);
        goto LABEL_47;
      }
      goto LABEL_31;
    }
    v19 = 4097;
    (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 80))(&v18, v6, a3, &v19);
    v15 = v18 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_90;
    goto LABEL_46;
  }
  if ( v7 == 10 )
  {
    v19 = 10;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 176))(
      &v18,
      v6,
      a3,
      &v19);
    v13 = v18 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v18 = 0;
      v17 = v13 | 1;
      sub_9C66B0((__int64 *)&v18);
    }
    else
    {
      v18 = 0;
      sub_9C66B0((__int64 *)&v18);
      v17 = 1;
    }
    v14 = v22;
    if ( v22 )
      goto LABEL_41;
    goto LABEL_42;
  }
  if ( v7 != 14 )
    goto LABEL_31;
  LODWORD(v17) = 14;
  (*(void (__fastcall **)(unsigned __int64 *, __int64 *, _QWORD *, unsigned __int64 *))(*v6 + 104))(&v19, v6, a3, &v17);
  if ( (v19 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v19 = 0;
    sub_9C66B0((__int64 *)&v19);
    v18 = 1;
    sub_9C66B0((__int64 *)&v19);
    goto LABEL_88;
  }
  v9 = &v18;
  v18 = 0;
  v19 = v19 & 0xFFFFFFFFFFFFFFFELL | 1;
  sub_9C6670((__int64 *)&v18, &v19);
  sub_9C66B0((__int64 *)&v19);
  if ( (v18 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
LABEL_88:
    v18 = 0;
    sub_9C66B0((__int64 *)&v18);
    goto LABEL_33;
  }
  v18 = v18 & 0xFFFFFFFFFFFFFFFELL | 1;
LABEL_26:
  *a1 = 0;
  sub_9C6670(a1, v9);
  sub_9C66B0((__int64 *)v9);
  return a1;
}
