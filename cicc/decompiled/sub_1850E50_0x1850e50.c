// Function: sub_1850E50
// Address: 0x1850e50
//
__int64 __fastcall sub_1850E50(__int64 a1)
{
  __int64 **v1; // rax
  __int64 **i; // rdx
  __int64 v3; // r13
  unsigned int v4; // r15d
  _BYTE *v5; // rbx
  _BYTE *v6; // r12
  __int64 v7; // r13
  __int64 v8; // r14
  unsigned __int64 v9; // rax
  unsigned __int8 v10; // cl
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  _QWORD v17[2]; // [rsp+10h] [rbp-130h] BYREF
  __int64 v18; // [rsp+20h] [rbp-120h]
  __int64 v19; // [rsp+28h] [rbp-118h]
  __int64 v20; // [rsp+30h] [rbp-110h]
  __int64 v21; // [rsp+38h] [rbp-108h]
  __int64 v22; // [rsp+40h] [rbp-100h]
  __int64 v23; // [rsp+48h] [rbp-F8h]
  __int64 **v24; // [rsp+50h] [rbp-F0h]
  __int64 **v25; // [rsp+58h] [rbp-E8h]
  __int64 v26; // [rsp+60h] [rbp-E0h]
  __int64 v27; // [rsp+68h] [rbp-D8h]
  __int64 v28; // [rsp+70h] [rbp-D0h]
  __int64 v29; // [rsp+78h] [rbp-C8h]
  _BYTE *v30; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+88h] [rbp-B8h]
  _BYTE v32[176]; // [rsp+90h] [rbp-B0h] BYREF

  v30 = v32;
  v31 = 0x1000000000LL;
  v17[0] = 0;
  v17[1] = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  sub_13C69A0((__int64)v17, a1);
  sub_13C6E30((__int64)v17);
  v1 = v25;
  for ( i = v24; v25 != v24; i = v24 )
  {
    if ( (char *)v1 - (char *)i == 8 )
    {
      v3 = **i;
      if ( v3 )
      {
        if ( !sub_15E4F60(**i) && !(unsigned __int8)sub_1560180(v3 + 112, 27) && (*(_BYTE *)(v3 + 32) & 0xF) == 7 )
        {
          v16 = (unsigned int)v31;
          if ( (unsigned int)v31 >= HIDWORD(v31) )
          {
            sub_16CD150((__int64)&v30, v32, 0, 8, v14, v15);
            v16 = (unsigned int)v31;
          }
          *(_QWORD *)&v30[8 * v16] = v3;
          LODWORD(v31) = v31 + 1;
        }
      }
    }
    sub_13C6E30((__int64)v17);
    v1 = v25;
  }
  if ( v27 )
    j_j___libc_free_0(v27, v29 - v27);
  if ( v24 )
    j_j___libc_free_0(v24, v26 - (_QWORD)v24);
  if ( v21 )
    j_j___libc_free_0(v21, v23 - v21);
  v4 = 0;
  j___libc_free_0(v18);
  v5 = v30;
  v6 = &v30[8 * (unsigned int)v31];
  if ( v30 != v6 )
  {
    while ( 1 )
    {
      v7 = *((_QWORD *)v6 - 1);
      v8 = *(_QWORD *)(v7 + 8);
      if ( v8 )
        break;
LABEL_23:
      if ( (unsigned __int8)sub_1560180(v7 + 112, 27) )
      {
LABEL_15:
        v6 -= 8;
        if ( v5 == v6 )
          goto LABEL_25;
      }
      else
      {
        v6 -= 8;
        sub_15E0D50(v7, -1, 27);
        v4 = 1;
        if ( v5 == v6 )
        {
LABEL_25:
          v6 = v30;
          goto LABEL_26;
        }
      }
    }
    while ( 1 )
    {
      v9 = (unsigned __int64)sub_1648700(v8);
      v10 = *(_BYTE *)(v9 + 16);
      if ( v10 <= 0x17u )
        goto LABEL_15;
      if ( v10 == 78 )
      {
        v11 = v9 | 4;
      }
      else
      {
        if ( v10 != 29 )
          goto LABEL_15;
        v11 = v9 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v12 || !(unsigned __int8)sub_1560180(*(_QWORD *)(*(_QWORD *)(v12 + 40) + 56LL) + 112LL, 27) )
        goto LABEL_15;
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        goto LABEL_23;
    }
  }
LABEL_26:
  if ( v6 != v32 )
    _libc_free((unsigned __int64)v6);
  return v4;
}
