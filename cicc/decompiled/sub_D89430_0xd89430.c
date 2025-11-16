// Function: sub_D89430
// Address: 0xd89430
//
__int64 __fastcall sub_D89430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  unsigned int v6; // eax
  unsigned int v7; // eax
  unsigned int v9; // eax
  unsigned __int64 v12; // r8
  __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned __int64 v17; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-78h]
  __int64 v19; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-68h]
  unsigned __int64 v21; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-58h]
  __int64 v23[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v24[8]; // [rsp+60h] [rbp-40h] BYREF

  if ( !a6 )
  {
    v9 = *(_DWORD *)(a2 + 24);
    v18 = v9;
    if ( v9 > 0x40 )
    {
      sub_C43690((__int64)&v17, a5, 1);
      v9 = v18;
      v13 = 1LL << ((unsigned __int8)v18 - 1);
      if ( v18 > 0x40 )
      {
        if ( (*(_QWORD *)(v17 + 8LL * ((v18 - 1) >> 6)) & v13) == 0 )
        {
          v22 = v18;
          sub_C43780((__int64)&v21, (const void **)&v17);
          goto LABEL_12;
        }
LABEL_17:
        v14 = *(_DWORD *)(a2 + 40);
        *(_DWORD *)(a1 + 8) = v14;
        if ( v14 > 0x40 )
          sub_C43780(a1, (const void **)(a2 + 32));
        else
          *(_QWORD *)a1 = *(_QWORD *)(a2 + 32);
        v15 = *(_DWORD *)(a2 + 56);
        *(_DWORD *)(a1 + 24) = v15;
        if ( v15 > 0x40 )
          sub_C43780(a1 + 16, (const void **)(a2 + 48));
        else
          *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 48);
        goto LABEL_21;
      }
    }
    else
    {
      v12 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & a5;
      if ( !v9 )
        v12 = 0;
      v13 = 1LL << ((unsigned __int8)v9 - 1);
      v17 = v12;
    }
    if ( (v13 & v17) == 0 )
    {
      v22 = v9;
      v21 = v17;
LABEL_12:
      v20 = *(_DWORD *)(a2 + 24);
      if ( v20 > 0x40 )
        sub_C43690((__int64)&v19, 0, 0);
      else
        v19 = 0;
      sub_AADC30((__int64)v23, (__int64)&v19, (__int64 *)&v21);
      sub_D89240(a1, a2, a3, a4, (__int64)v23);
      sub_969240(v24);
      sub_969240(v23);
      sub_969240(&v19);
      sub_969240((__int64 *)&v21);
LABEL_21:
      sub_969240((__int64 *)&v17);
      return a1;
    }
    goto LABEL_17;
  }
  v6 = *(_DWORD *)(a2 + 40);
  *(_DWORD *)(a1 + 8) = v6;
  if ( v6 > 0x40 )
  {
    sub_C43780(a1, (const void **)(a2 + 32));
    v16 = *(_DWORD *)(a2 + 56);
    *(_DWORD *)(a1 + 24) = v16;
    if ( v16 <= 0x40 )
      goto LABEL_4;
LABEL_23:
    sub_C43780(a1 + 16, (const void **)(a2 + 48));
    return a1;
  }
  *(_QWORD *)a1 = *(_QWORD *)(a2 + 32);
  v7 = *(_DWORD *)(a2 + 56);
  *(_DWORD *)(a1 + 24) = v7;
  if ( v7 > 0x40 )
    goto LABEL_23;
LABEL_4:
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 48);
  return a1;
}
