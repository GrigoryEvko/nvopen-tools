// Function: sub_2C37F10
// Address: 0x2c37f10
//
void __fastcall sub_2C37F10(__int64 *a1)
{
  __int64 v1; // rsi
  __int64 v2; // r13
  __int64 v3; // rdx
  __int64 v4; // r12
  _QWORD *v5; // rbx
  __int64 *v6; // r15
  unsigned __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // [rsp+10h] [rbp-170h]
  __int64 v14; // [rsp+18h] [rbp-168h]
  _BYTE *v15; // [rsp+20h] [rbp-160h] BYREF
  char v16; // [rsp+29h] [rbp-157h]
  unsigned __int64 v17; // [rsp+30h] [rbp-150h]
  char v18; // [rsp+39h] [rbp-147h]
  _BYTE *v19; // [rsp+40h] [rbp-140h] BYREF
  __int64 v20; // [rsp+48h] [rbp-138h]
  _BYTE v21[64]; // [rsp+50h] [rbp-130h] BYREF
  __int64 v22[12]; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v23[6]; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v24; // [rsp+120h] [rbp-60h]

  v1 = *a1;
  v19 = v21;
  v20 = 0x800000000LL;
  sub_2C363F0((__int64)&v19, v1);
  v16 = 1;
  v17 = (unsigned __int64)v19;
  v15 = &v19[8 * (unsigned int)v20];
  v18 = 1;
  sub_2C26110((__int64)v23, (__int64 *)&v15);
  sub_2C25F40((__int64)v22, v23);
  sub_2C29250((__int64)v23, v22);
  v2 = v23[0];
  v13 = v24;
  while ( v2 != v13 )
  {
    v3 = *(_QWORD *)v2;
    v14 = v2 + 8;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)v2 + 8LL) - 1 > 1 )
    {
      v10 = (__int64 *)(v2 + 8);
      do
        v11 = *v10++;
      while ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 1 > 1 );
      v4 = v11 + 112;
      v5 = (_QWORD *)(*(_QWORD *)(v11 + 112) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v5 != (_QWORD *)(v11 + 112) )
      {
        do
        {
LABEL_6:
          while ( 1 )
          {
            v6 = v5 - 3;
            v7 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
            v8 = (__int64)(v5 - 3);
            v5 = (_QWORD *)v7;
            if ( (unsigned __int8)sub_2C253E0(v8) )
              break;
            if ( v7 == v4 )
              goto LABEL_8;
          }
          sub_2C19E60(v6);
        }
        while ( v7 != v4 );
LABEL_8:
        v9 = *(_QWORD *)v2;
        v2 += 8;
        if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 1 <= 1 )
          continue;
      }
      v2 = v14;
      do
      {
        v12 = *(_QWORD *)v2;
        v2 += 8;
      }
      while ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 1 > 1 );
    }
    else
    {
      v4 = v3 + 112;
      v5 = (_QWORD *)(*(_QWORD *)(v3 + 112) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v5 != (_QWORD *)(v3 + 112) )
        goto LABEL_6;
      v2 += 8;
    }
  }
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
}
