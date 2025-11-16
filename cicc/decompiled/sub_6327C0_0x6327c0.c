// Function: sub_6327C0
// Address: 0x6327c0
//
__int64 __fastcall sub_6327C0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  char v12; // al
  __int64 v13; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  char v20; // al
  __int64 v21; // r14
  __int64 v22; // [rsp-8h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-48h] BYREF
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  __int64 v25; // [rsp+18h] [rbp-38h]

  v8 = a2;
  v23 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( (*(_BYTE *)(a1 + 145) & 0x20) != 0 )
  {
    sub_5F8060(a1, a3, v10, v11);
    if ( (*(_BYTE *)(a1 + 145) & 0x40) != 0 )
    {
      v8 = sub_740B80(*(_QWORD *)(a1 + 152), (a4[42] & 0x10) == 0 ? 0x20000 : 131200);
      if ( v8 )
        goto LABEL_3;
      goto LABEL_28;
    }
    v8 = *(_QWORD *)(a1 + 152);
  }
  if ( v8 )
  {
LABEL_3:
    v24 = 0;
    v25 = 0;
    goto LABEL_4;
  }
LABEL_28:
  v20 = a4[40];
  a4[41] |= 2u;
  if ( (v20 & 0x20) == 0 )
  {
    sub_685360(2420, a5);
    v20 = a4[40];
  }
  if ( (v20 & 0x40) != 0 )
  {
    v24 = 0;
    v13 = 0;
    v25 = 0;
    goto LABEL_12;
  }
  v24 = 0;
  v25 = 0;
  v8 = sub_72C9D0();
  if ( !v8 )
    goto LABEL_34;
LABEL_4:
  if ( (unsigned int)sub_7A1C60(v8, a5, *(_QWORD *)(a1 + 120), 0, v23, (unsigned int)&v24, 0)
    && (unsigned int)sub_71ACC0(v23) )
  {
    if ( (*(_BYTE *)(v23 + 170) & 0x40) != 0 )
      a4[41] |= 0x10u;
    if ( (a4[40] & 0x40) == 0 )
    {
      v13 = sub_724E50(&v23, v22, v17, v18, v19);
      goto LABEL_12;
    }
LABEL_34:
    v13 = 0;
    goto LABEL_12;
  }
  v12 = a4[40];
  if ( (v12 & 4) != 0 && !word_4D04898 )
  {
    if ( (v12 & 0x20) == 0 )
    {
      v21 = sub_67E020(2639, a5, *(_QWORD *)a1);
      sub_67E370(v21, &v24);
      sub_685910(v21);
      v12 = a4[40];
    }
    a4[41] |= 2u;
  }
  v13 = 0;
  if ( (v12 & 0x40) == 0 )
  {
    if ( *(char *)(a1 + 144) >= 0 )
    {
      v15 = sub_740B80(v8, (a4[42] & 0x10) == 0 ? 131104 : 131232);
      v8 = v15;
      if ( *(_QWORD *)(v15 + 16) )
        sub_734250(v15, ((a4[42] >> 4) ^ 1) & 1);
    }
    v16 = sub_724D50(9);
    *(_QWORD *)(v16 + 176) = v8;
    v13 = v16;
    if ( *(char *)(v8 + 50) < 0 )
    {
      a4[41] |= 0x10u;
      *(_BYTE *)(v16 + 170) |= 0x40u;
    }
    *(_QWORD *)(v16 + 128) = *(_QWORD *)(a1 + 120);
  }
  a4[41] |= 4u;
LABEL_12:
  sub_67E3D0(&v24);
  if ( v23 )
    sub_724E30(&v23);
  return v13;
}
