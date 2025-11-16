// Function: sub_22CD2E0
// Address: 0x22cd2e0
//
__int64 __fastcall sub_22CD2E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // rcx
  __int64 v8; // rcx
  bool v10; // zf
  __int64 v12; // [rsp+10h] [rbp-E0h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-D8h]
  __int64 v14; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v16[48]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v17; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v18; // [rsp+68h] [rbp-88h]
  __int64 v19; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v20; // [rsp+78h] [rbp-78h]
  char v21; // [rsp+80h] [rbp-70h]
  unsigned __int8 v22[8]; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v23; // [rsp+98h] [rbp-58h]
  unsigned int v24; // [rsp+A0h] [rbp-50h]
  unsigned __int64 v25; // [rsp+A8h] [rbp-48h]
  unsigned int v26; // [rsp+B0h] [rbp-40h]
  char v27; // [rsp+B8h] [rbp-38h]

  *(_QWORD *)v16 = 0;
  if ( sub_AA5B70(a4) )
  {
    sub_B2D8F0((__int64)&v17, a3);
    if ( v21 )
    {
      v13 = v18;
      if ( v18 > 0x40 )
        sub_C43780((__int64)&v12, (const void **)&v17);
      else
        v12 = v17;
      v15 = v20;
      if ( v20 > 0x40 )
        sub_C43780((__int64)&v14, (const void **)&v19);
      else
        v14 = v19;
      sub_22C06B0((__int64)v22, (__int64)&v12, 0);
      sub_22C0650(a1, v22);
      *(_BYTE *)(a1 + 40) = 1;
      sub_22C0090(v22);
      sub_969240(&v14);
      sub_969240(&v12);
      if ( v21 )
      {
        v21 = 0;
        sub_969240(&v19);
        sub_969240(&v17);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 40) = 1;
      *(_WORD *)a1 = 6;
      *(_WORD *)v22 = 0;
      sub_22C0090(v22);
    }
    goto LABEL_19;
  }
  v6 = *(_QWORD *)(a4 + 16);
  if ( !v6 )
  {
LABEL_18:
    sub_22C0650(a1, v16);
    *(_BYTE *)(a1 + 40) = 1;
    goto LABEL_19;
  }
  while ( 1 )
  {
    v7 = *(_QWORD *)(v6 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
      break;
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      goto LABEL_18;
  }
LABEL_13:
  v8 = *(_QWORD *)(v7 + 40);
  if ( a4 == v8 )
    goto LABEL_17;
  sub_22CCF60((__int64)v22, a2, a3, v8, a4, 0);
  if ( !v27 )
  {
    *(_BYTE *)(a1 + 40) = 0;
    goto LABEL_19;
  }
  sub_22C0C70((__int64)v16, (__int64)v22, 0, 0, 1u);
  if ( v16[0] != 6 )
  {
    if ( v27 )
    {
      v27 = 0;
      if ( (unsigned int)v22[0] - 4 <= 1 )
      {
        if ( v26 > 0x40 && v25 )
          j_j___libc_free_0_0(v25);
        if ( v24 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
      }
    }
LABEL_17:
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_18;
      v7 = *(_QWORD *)(v6 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
        goto LABEL_13;
    }
  }
  sub_22C0650(a1, v16);
  v10 = v27 == 0;
  *(_BYTE *)(a1 + 40) = 1;
  if ( !v10 )
  {
    v27 = 0;
    sub_22C0090(v22);
  }
LABEL_19:
  sub_22C0090(v16);
  return a1;
}
