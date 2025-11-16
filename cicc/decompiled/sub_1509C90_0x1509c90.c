// Function: sub_1509C90
// Address: 0x1509c90
//
__int64 *__fastcall sub_1509C90(__int64 *a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 i; // rbx
  __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // r14
  __int64 v13; // rdi
  _QWORD *v14; // rbx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // r15
  __int64 v19; // rdi
  __int64 *v20; // rbx
  __int64 v21; // rsi
  __int64 v22[2]; // [rsp+10h] [rbp-50h] BYREF
  char v23; // [rsp+20h] [rbp-40h]
  char v24; // [rsp+21h] [rbp-3Fh]

  sub_14EB830((unsigned __int64 *)v22, a2);
  v4 = v22[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v22[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_13;
  v5 = *(_QWORD *)(a2 + 440);
  *(_BYTE *)(a2 + 1657) = 1;
  v6 = *(_QWORD *)(v5 + 32);
  for ( i = v5 + 24; i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v8 = v6 - 56;
    if ( !v6 )
      v8 = 0;
    sub_1503DC0((unsigned __int64 *)v22, a2, v8);
    v4 = v22[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v22[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_13;
  }
  v9 = *(_QWORD *)(a2 + 456);
  if ( *(_OWORD *)(a2 + 448) != 0 )
  {
    if ( v9 < *(_QWORD *)(a2 + 448) )
      v9 = *(_QWORD *)(a2 + 448);
    sub_1505110(v22, a2, v9, 0);
    v4 = v22[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v22[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_13:
      *a1 = v4 | 1;
      return a1;
    }
  }
  if ( *(_DWORD *)(a2 + 1560) )
  {
    v24 = 1;
    v23 = 3;
    v22[0] = (__int64)"Never resolved function from blockaddress";
    sub_14EE4B0(a1, a2 + 8, (__int64)v22);
  }
  else
  {
    if ( *(_DWORD *)(a2 + 1432) )
    {
      v11 = *(_QWORD **)(a2 + 1424);
      v12 = &v11[2 * *(unsigned int *)(a2 + 1440)];
      if ( v11 != v12 )
      {
        while ( 1 )
        {
          v13 = *v11;
          v14 = v11;
          if ( *v11 != -16 && v13 != -8 )
            break;
          v11 += 2;
          if ( v12 == v11 )
            goto LABEL_16;
        }
        if ( v11 != v12 )
        {
          while ( 1 )
          {
            v15 = *(_QWORD *)(v13 + 8);
            if ( v15 )
            {
              do
              {
                while ( 1 )
                {
                  v16 = sub_1648700(v15);
                  if ( *(_BYTE *)(v16 + 16) == 78 )
                    break;
                  v15 = *(_QWORD *)(v15 + 8);
                  if ( !v15 )
                    goto LABEL_29;
                }
                sub_156E800(v16, v14[1]);
                v15 = *(_QWORD *)(v15 + 8);
              }
              while ( v15 );
LABEL_29:
              if ( *(_QWORD *)(*v14 + 8LL) )
                sub_164D160(*v14, v14[1]);
            }
            ((void (*)(void))sub_15E3D00)();
            v14 += 2;
            if ( v14 == v12 )
              break;
            while ( *v14 == -8 || *v14 == -16 )
            {
              v14 += 2;
              if ( v12 == v14 )
                goto LABEL_16;
            }
            if ( v14 == v12 )
              break;
            v13 = *v14;
          }
        }
      }
    }
LABEL_16:
    sub_14EF490(a2 + 1416);
    if ( *(_DWORD *)(a2 + 1464) )
    {
      v17 = *(__int64 **)(a2 + 1456);
      v18 = &v17[2 * *(unsigned int *)(a2 + 1472)];
      if ( v17 != v18 )
      {
        while ( 1 )
        {
          v19 = *v17;
          v20 = v17;
          if ( *v17 != -16 && v19 != -8 )
            break;
          v17 += 2;
          if ( v18 == v17 )
            goto LABEL_17;
        }
        if ( v17 != v18 )
        {
          while ( 1 )
          {
            v21 = v20[1];
            v20 += 2;
            sub_164D160(v19, v21);
            sub_15E3D00(*(v20 - 2));
            if ( v20 == v18 )
              break;
            while ( *v20 == -8 || *v20 == -16 )
            {
              v20 += 2;
              if ( v18 == v20 )
                goto LABEL_17;
            }
            if ( v18 == v20 )
              break;
            v19 = *v20;
          }
        }
      }
    }
LABEL_17:
    sub_14EF490(a2 + 1448);
    sub_157E370(*(_QWORD *)(a2 + 440));
    sub_1569750(*(_QWORD *)(a2 + 440));
    sub_1569190(*(_QWORD *)(a2 + 440));
    *a1 = 1;
  }
  return a1;
}
