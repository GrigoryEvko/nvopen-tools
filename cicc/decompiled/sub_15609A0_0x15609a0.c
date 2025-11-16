// Function: sub_15609A0
// Address: 0x15609a0
//
__int64 __fastcall sub_15609A0(__int64 *a1, _QWORD *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // r8
  __int64 v4; // rax
  __int64 v5; // r15
  _QWORD *v6; // rax
  int v7; // ebx
  _QWORD *v8; // r8
  char *v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // r12
  _QWORD *v13; // [rsp+0h] [rbp-A0h]
  __int64 v14; // [rsp+8h] [rbp-98h]
  unsigned int v15; // [rsp+14h] [rbp-8Ch] BYREF
  unsigned int v16; // [rsp+18h] [rbp-88h] BYREF
  char *v17; // [rsp+20h] [rbp-80h] BYREF
  __int64 v18; // [rsp+28h] [rbp-78h]
  _BYTE v19[112]; // [rsp+30h] [rbp-70h] BYREF

  v2 = 0;
  v17 = v19;
  v18 = 0x800000000LL;
  do
  {
    if ( (*a2 & (1LL << v2)) != 0 )
    {
      if ( v2 == 9 )
      {
        v3 = sub_155D350(a1, a2[9]);
      }
      else if ( v2 <= 9 )
      {
        if ( v2 != 1 )
        {
          if ( v2 == 2 )
          {
            sub_1560970((__int64)&v15, (__int64)a2);
            v3 = sub_155D370(a1, v15, &v16);
            goto LABEL_5;
          }
          goto LABEL_15;
        }
        v3 = sub_155D330(a1, a2[7]);
      }
      else
      {
        if ( v2 != 10 )
        {
          if ( v2 == 48 )
          {
            v3 = sub_155D340(a1, a2[8]);
            v4 = (unsigned int)v18;
            if ( (unsigned int)v18 >= HIDWORD(v18) )
              goto LABEL_14;
            goto LABEL_6;
          }
LABEL_15:
          v3 = sub_155CEC0(a1, v2, 0);
          goto LABEL_5;
        }
        v3 = sub_155D360(a1, a2[10]);
      }
LABEL_5:
      v4 = (unsigned int)v18;
      if ( (unsigned int)v18 >= HIDWORD(v18) )
      {
LABEL_14:
        v14 = v3;
        sub_16CD150(&v17, v19, 0, 8);
        v4 = (unsigned int)v18;
        v3 = v14;
      }
LABEL_6:
      *(_QWORD *)&v17[8 * v4] = v3;
      LODWORD(v18) = v18 + 1;
    }
    ++v2;
  }
  while ( v2 != 59 );
  v5 = a2[4];
  if ( a2 + 2 == (_QWORD *)v5 )
  {
    v10 = v18;
  }
  else
  {
    do
    {
      v6 = sub_155D020(a1, *(_BYTE **)(v5 + 32), *(_QWORD *)(v5 + 40), *(_BYTE **)(v5 + 64), *(_QWORD *)(v5 + 72));
      v7 = v18;
      v8 = v6;
      if ( (unsigned int)v18 >= HIDWORD(v18) )
      {
        v13 = v6;
        sub_16CD150(&v17, v19, 0, 8);
        v7 = v18;
        v8 = v13;
      }
      v9 = &v17[8 * v7];
      if ( v9 )
      {
        *(_QWORD *)v9 = v8;
        v7 = v18;
      }
      v10 = v7 + 1;
      LODWORD(v18) = v10;
      v5 = sub_220EF30(v5);
    }
    while ( a2 + 2 != (_QWORD *)v5 );
  }
  v11 = sub_155EF10(a1, v17, v10);
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v17);
  return v11;
}
