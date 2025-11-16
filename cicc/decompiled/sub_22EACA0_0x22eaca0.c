// Function: sub_22EACA0
// Address: 0x22eaca0
//
__int64 __fastcall sub_22EACA0(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  unsigned __int8 v3; // al
  unsigned __int8 v5; // dl
  unsigned __int8 v7; // al
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  char v12; // dl
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+8h] [rbp-58h]
  unsigned __int64 v17; // [rsp+10h] [rbp-50h]
  unsigned int v18; // [rsp+18h] [rbp-48h]
  unsigned __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-38h]
  unsigned __int64 v21; // [rsp+30h] [rbp-30h]
  unsigned int v22; // [rsp+38h] [rbp-28h]

  v3 = *a2;
  if ( !*a2 || (v5 = *a3) == 0 )
  {
    *(_WORD *)a1 = 0;
    return a1;
  }
  if ( v3 == 6 )
  {
    *(_BYTE *)a1 = v5;
    *(_BYTE *)(a1 + 1) = 0;
    if ( v5 > 3u )
    {
      if ( (unsigned __int8)(v5 - 4) > 1u )
        return a1;
      goto LABEL_28;
    }
    if ( v5 == 1 )
      return a1;
LABEL_16:
    *(_QWORD *)(a1 + 8) = *((_QWORD *)a3 + 1);
    return a1;
  }
  if ( v5 == 6 )
  {
    *(_BYTE *)a1 = v3;
    *(_BYTE *)(a1 + 1) = 0;
    if ( v3 <= 3u )
    {
      if ( v3 == 1 )
        return a1;
      goto LABEL_8;
    }
LABEL_17:
    if ( (unsigned __int8)(v3 - 4) > 1u )
      return a1;
    goto LABEL_18;
  }
  if ( sub_22EAA60((__int64)a2) )
  {
    v3 = *a2;
    *(_WORD *)a1 = *a2;
    if ( v3 <= 3u )
      goto LABEL_7;
    goto LABEL_17;
  }
  if ( sub_22EAA60((__int64)a3) )
  {
    v7 = *a3;
    *(_WORD *)a1 = *a3;
    if ( v7 > 3u )
    {
      if ( (unsigned __int8)(v7 - 4) > 1u )
        return a1;
LABEL_28:
      v10 = *((_DWORD *)a3 + 4);
      *(_DWORD *)(a1 + 16) = v10;
      if ( v10 > 0x40 )
        sub_C43780(a1 + 8, (const void **)a3 + 1);
      else
        *(_QWORD *)(a1 + 8) = *((_QWORD *)a3 + 1);
      v11 = *((_DWORD *)a3 + 8);
      *(_DWORD *)(a1 + 32) = v11;
      if ( v11 > 0x40 )
        sub_C43780(a1 + 24, (const void **)a3 + 3);
      else
        *(_QWORD *)(a1 + 24) = *((_QWORD *)a3 + 3);
      *(_BYTE *)(a1 + 1) = a3[1];
      return a1;
    }
    if ( v7 <= 1u )
      return a1;
    goto LABEL_16;
  }
  v3 = *a2;
  if ( (unsigned __int8)(*a2 - 4) <= 1u )
  {
    if ( (unsigned __int8)(*a3 - 4) <= 1u )
    {
      sub_AB2160((__int64)&v15, (__int64)(a2 + 8), (__int64)(a3 + 8), 0);
      v12 = 1;
      if ( *a2 != 5 )
        v12 = *a3 == 5;
      v13 = v16;
      v16 = 0;
      v20 = v13;
      v19 = v15;
      v14 = v18;
      v18 = 0;
      v22 = v14;
      v21 = v17;
      sub_22C06B0(a1, (__int64)&v19, v12);
      if ( v22 > 0x40 && v21 )
        j_j___libc_free_0_0(v21);
      if ( v20 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
      return a1;
    }
    *(_BYTE *)a1 = v3;
    *(_BYTE *)(a1 + 1) = 0;
LABEL_18:
    v8 = *((_DWORD *)a2 + 4);
    *(_DWORD *)(a1 + 16) = v8;
    if ( v8 > 0x40 )
      sub_C43780(a1 + 8, (const void **)a2 + 1);
    else
      *(_QWORD *)(a1 + 8) = *((_QWORD *)a2 + 1);
    v9 = *((_DWORD *)a2 + 8);
    *(_DWORD *)(a1 + 32) = v9;
    if ( v9 > 0x40 )
      sub_C43780(a1 + 24, (const void **)a2 + 3);
    else
      *(_QWORD *)(a1 + 24) = *((_QWORD *)a2 + 3);
    *(_BYTE *)(a1 + 1) = a2[1];
    return a1;
  }
  *(_BYTE *)a1 = v3;
  *(_BYTE *)(a1 + 1) = 0;
  if ( v3 > 3u )
    return a1;
LABEL_7:
  if ( v3 > 1u )
LABEL_8:
    *(_QWORD *)(a1 + 8) = *((_QWORD *)a2 + 1);
  return a1;
}
