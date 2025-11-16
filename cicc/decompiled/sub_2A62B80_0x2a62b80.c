// Function: sub_2A62B80
// Address: 0x2a62b80
//
_WORD *__fastcall sub_2A62B80(_WORD *a1, unsigned __int8 *a2)
{
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  unsigned __int8 v5; // al
  __int64 v6; // rsi
  char v7; // si
  unsigned __int8 *v8; // rax
  __int64 v10; // rsi
  __int64 v11; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-78h]
  __int64 v13; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-68h]
  __int64 v15; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-58h]
  __int64 v17; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-48h]
  char v19; // [rsp+40h] [rbp-40h]

  v3 = *((_QWORD *)a2 + 1);
  v4 = (unsigned int)*a2 - 34;
  v5 = *(_BYTE *)(v3 + 8);
  if ( (unsigned __int8)(*a2 - 34) > 0x33u )
    goto LABEL_7;
  v6 = 0x8000000000041LL;
  if ( !_bittest64(&v6, v4) )
    goto LABEL_7;
  v7 = *(_BYTE *)(v3 + 8);
  if ( (unsigned int)v5 - 17 <= 1 )
    v7 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
  if ( v7 != 12 )
  {
LABEL_6:
    if ( v5 == 14 )
    {
      if ( (unsigned __int8)sub_B493B0(a2) )
      {
LABEL_12:
        v8 = (unsigned __int8 *)sub_AC9EC0(*((__int64 ***)a2 + 1));
        *a1 = 0;
        sub_2A62A00((__int64)a1, v8);
        return a1;
      }
      v3 = *((_QWORD *)a2 + 1);
      v5 = *(_BYTE *)(v3 + 8);
    }
LABEL_7:
    if ( (unsigned int)v5 - 17 <= 1 )
      v5 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
    if ( v5 == 12 )
    {
      if ( (a2[7] & 0x20) == 0 )
        goto LABEL_14;
      v10 = sub_B91C10((__int64)a2, 4);
      if ( v10 )
      {
        sub_ABEA30((__int64)&v15, v10);
        sub_22C06B0((__int64)a1, (__int64)&v15, 0);
        sub_969240(&v17);
        sub_969240(&v15);
        return a1;
      }
    }
    if ( (a2[7] & 0x20) != 0 && sub_B91C10((__int64)a2, 11) )
      goto LABEL_12;
LABEL_14:
    *a1 = 6;
    return a1;
  }
  sub_B492D0((__int64)&v15, (__int64)a2);
  if ( !v19 )
  {
    v3 = *((_QWORD *)a2 + 1);
    v5 = *(_BYTE *)(v3 + 8);
    goto LABEL_6;
  }
  v12 = v16;
  if ( v16 > 0x40 )
    sub_C43780((__int64)&v11, (const void **)&v15);
  else
    v11 = v15;
  v14 = v18;
  if ( v18 > 0x40 )
    sub_C43780((__int64)&v13, (const void **)&v17);
  else
    v13 = v17;
  sub_22C06B0((__int64)a1, (__int64)&v11, 0);
  sub_969240(&v13);
  sub_969240(&v11);
  if ( v19 )
  {
    v19 = 0;
    sub_969240(&v17);
    sub_969240(&v15);
  }
  return a1;
}
