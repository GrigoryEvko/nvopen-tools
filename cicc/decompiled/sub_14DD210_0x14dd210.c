// Function: sub_14DD210
// Address: 0x14dd210
//
__int64 __fastcall sub_14DD210(__int64 *a1, _BYTE *a2, __int64 a3)
{
  __int64 *v3; // r15
  int v6; // eax
  char v7; // cl
  char v8; // di
  __int64 v9; // rdx
  char *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rbx
  unsigned __int8 v16; // al
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r15
  int v20; // ebx
  __int64 *v22; // rsi
  signed __int64 v23; // rdx
  __int64 *v24; // rax
  __int64 **v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // [rsp+0h] [rbp-100h]
  __int64 v32; // [rsp+8h] [rbp-F8h]
  __int64 *v33; // [rsp+10h] [rbp-F0h]
  __int64 v34; // [rsp+18h] [rbp-E8h]
  __int64 v36; // [rsp+28h] [rbp-D8h]
  __int64 v37; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-C8h]
  __int64 v39; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v40; // [rsp+80h] [rbp-80h] BYREF
  __int64 v41; // [rsp+88h] [rbp-78h]
  _QWORD v42[8]; // [rsp+90h] [rbp-70h] BYREF
  char v43; // [rsp+D0h] [rbp-30h] BYREF

  v3 = a1;
  v6 = *((_DWORD *)a1 + 5);
  v7 = *((_BYTE *)a1 + 16);
  v8 = *((_BYTE *)a1 + 23) & 0x40;
  v9 = 24LL * (v6 & 0xFFFFFFF);
  if ( v7 != 77 )
  {
    v22 = &a1[v9 / 0xFFFFFFFFFFFFFFF8LL];
    if ( v8 )
    {
      v22 = (__int64 *)*(a1 - 1);
      v3 = &v22[(unsigned __int64)v9 / 8];
    }
    v23 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 3);
    if ( v23 >> 2 )
    {
      v24 = v22;
      while ( *(_BYTE *)(*v24 + 16) <= 0x10u )
      {
        if ( *(_BYTE *)(v24[3] + 16) > 0x10u )
        {
          v24 += 3;
          break;
        }
        if ( *(_BYTE *)(v24[6] + 16) > 0x10u )
        {
          v24 += 6;
          break;
        }
        if ( *(_BYTE *)(v24[9] + 16) > 0x10u )
        {
          v24 += 9;
          break;
        }
        v24 += 12;
        if ( &v22[12 * (v23 >> 2)] == v24 )
        {
          v23 = 0xAAAAAAAAAAAAAAABLL * (v3 - v24);
          goto LABEL_69;
        }
      }
LABEL_44:
      v36 = 0;
      if ( v24 != v3 )
        return v36;
      goto LABEL_45;
    }
    v24 = v22;
LABEL_69:
    if ( v23 != 2 )
    {
      if ( v23 != 3 )
      {
        if ( v23 != 1 )
          goto LABEL_45;
        goto LABEL_72;
      }
      if ( *(_BYTE *)(*v24 + 16) > 0x10u )
        goto LABEL_44;
      v24 += 3;
    }
    if ( *(_BYTE *)(*v24 + 16) > 0x10u )
      goto LABEL_44;
    v24 += 3;
LABEL_72:
    if ( *(_BYTE *)(*v24 + 16) > 0x10u )
      goto LABEL_44;
LABEL_45:
    v25 = (__int64 **)&v39;
    v37 = 0;
    v38 = 1;
    do
    {
      *v25 = (__int64 *)-8LL;
      v25 += 2;
    }
    while ( v25 != &v40 );
    v40 = v42;
    v41 = 0x800000000LL;
    if ( v22 != v3 )
    {
      v26 = v22;
      do
      {
        v27 = *v26;
        v28 = sub_14DB6D0(*v26, (__int64)a2, a3, (__int64)&v37);
        v29 = (unsigned int)v41;
        if ( !v28 )
          v28 = v27;
        if ( (unsigned int)v41 >= HIDWORD(v41) )
        {
          v32 = v28;
          sub_16CD150(&v40, v42, 0, 8);
          v29 = (unsigned int)v41;
          v28 = v32;
        }
        v26 += 3;
        v40[v29] = v28;
        LODWORD(v41) = v41 + 1;
      }
      while ( v3 != v26 );
      v7 = *((_BYTE *)a1 + 16);
    }
    if ( (unsigned __int8)(v7 - 75) > 1u )
    {
      switch ( v7 )
      {
        case '6':
          if ( (*((_BYTE *)a1 + 18) & 1) != 0 )
          {
            v36 = 0;
          }
          else
          {
            v36 = 0;
            v30 = *(a1 - 3);
            if ( *(_BYTE *)(v30 + 16) <= 0x10u )
              v36 = sub_14D8290(v30, *a1, a2);
          }
          break;
        case 'W':
          v36 = sub_15A3A20(*(a1 - 6), *(a1 - 3), a1[7], *((unsigned int *)a1 + 16), 0);
          break;
        case 'V':
          v36 = sub_1584BD0(*v40, a1[7], *((unsigned int *)a1 + 16));
          break;
        default:
          v36 = sub_14DD1F0((__int64)a1, v40, (unsigned int)v41, a2, a3);
          break;
      }
    }
    else
    {
      v36 = sub_14D7760(*((_WORD *)a1 + 9) & 0x7FFF, (_QWORD *)*v40, v40[1], (__int64)a2, a3);
    }
    if ( v40 != v42 )
      _libc_free((unsigned __int64)v40);
    if ( (v38 & 1) == 0 )
      j___libc_free_0(v39);
    return v36;
  }
  v40 = 0;
  v10 = (char *)v42;
  v41 = 1;
  do
  {
    *(_QWORD *)v10 = -8;
    v10 += 16;
  }
  while ( v10 != &v43 );
  v11 = *(_QWORD *)(*(_QWORD *)(a1[5] + 56) + 80LL);
  v12 = v11 - 24;
  if ( !v11 )
    v12 = 0;
  v13 = &a1[v9 / 0xFFFFFFFFFFFFFFF8LL];
  v31 = v12;
  if ( v8 )
    v13 = (__int64 *)*(a1 - 1);
  v34 = 0;
  v33 = &v13[(unsigned __int64)v9 / 8];
  v36 = 0;
  if ( &v13[(unsigned __int64)v9 / 8] == v13 )
    goto LABEL_75;
  do
  {
    v15 = *v13;
    v16 = *(_BYTE *)(*v13 + 16);
    if ( v16 != 9 )
    {
      if ( v16 <= 0x10u )
      {
        v14 = sub_14DB6D0(*v13, (__int64)a2, a3, (__int64)&v40);
        if ( !v14 )
          v14 = v15;
        if ( v36 && v14 != v36 )
          goto LABEL_28;
        v36 = v14;
      }
      else
      {
        if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
          v17 = (__int64 *)*(a1 - 1);
        else
          v17 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
        v18 = v17[3 * *((unsigned int *)a1 + 14) + 1 + v34];
        if ( v31 == v18 )
          goto LABEL_28;
        v19 = *(_QWORD *)(v18 + 8);
        if ( v19 )
        {
          while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v19) + 16) - 25) > 9u )
          {
            v19 = *(_QWORD *)(v19 + 8);
            if ( !v19 )
              goto LABEL_15;
          }
          v20 = 0;
          while ( 1 )
          {
            v19 = *(_QWORD *)(v19 + 8);
            if ( !v19 )
              break;
            while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v19) + 16) - 25) <= 9u )
            {
              v19 = *(_QWORD *)(v19 + 8);
              ++v20;
              if ( !v19 )
                goto LABEL_27;
            }
          }
LABEL_27:
          if ( v20 != -1 )
          {
LABEL_28:
            v36 = 0;
            goto LABEL_29;
          }
        }
      }
    }
LABEL_15:
    ++v34;
    v13 += 3;
  }
  while ( v33 != v13 );
  if ( !v36 )
LABEL_75:
    v36 = sub_1599EF0(*a1);
LABEL_29:
  if ( (v41 & 1) == 0 )
    j___libc_free_0(v42[0]);
  return v36;
}
