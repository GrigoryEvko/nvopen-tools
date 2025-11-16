// Function: sub_8BFF80
// Address: 0x8bff80
//
__int64 **__fastcall sub_8BFF80(unsigned __int64 a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v3; // r12
  char v5; // al
  __int64 v6; // r14
  _QWORD *v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 *v11; // r9
  __m128i *v12; // rdi
  __int64 v13; // rsi
  __int64 **v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // r9
  __int64 v19; // r9
  const __m128i *v20; // rdx
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // rax
  __int64 v26; // r15
  __m128i *v27; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1;
  v5 = *(_BYTE *)(a1 + 80);
  if ( v5 == 16 )
  {
    v3 = **(_QWORD **)(a1 + 88);
    v5 = *(_BYTE *)(v3 + 80);
  }
  if ( v5 == 24 )
  {
    v3 = *(_QWORD *)(v3 + 88);
    v5 = *(_BYTE *)(v3 + 80);
  }
  switch ( v5 )
  {
    case 4:
    case 5:
      v6 = *(_QWORD *)(*(_QWORD *)(v3 + 96) + 80LL);
      break;
    case 6:
      v6 = *(_QWORD *)(*(_QWORD *)(v3 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v6 = *(_QWORD *)(*(_QWORD *)(v3 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v6 = *(_QWORD *)(v3 + 88);
      break;
    default:
      BUG();
  }
  v7 = **(_QWORD ***)(v6 + 328);
  *a3 = (__int64)sub_8A3C00((__int64)v7, a2, 0, (__int64 *)(v3 + 48));
  sub_865900(v3);
  v12 = (__m128i *)*a3;
  if ( *a3 && (a2 = (__int64)v7, (unsigned int)sub_8AF210(v12, v7, 0, v3, v6, 0)) )
  {
    v13 = *a3;
    v14 = sub_8B1C20(v3, *a3, 0, v7, 0);
    sub_864110(v3, v13, v15, v16, v17, v18);
    if ( v14 )
    {
      v20 = (const __m128i *)*a3;
      if ( *(_BYTE *)(v3 + 80) != 20 || (*(_BYTE *)(*(_QWORD *)(v6 + 176) + 207LL) & 0x10) == 0 )
        goto LABEL_11;
      v27 = sub_72F240((const __m128i *)*a3);
      v25 = sub_8B74F0(v3, (__int64 ***)&v27, 1u, dword_4F07508, v23, v24);
      if ( !v25 || (unsigned __int8)(*((_BYTE *)v25 + 80) - 10) > 1u )
        sub_721090();
      v26 = v25[11];
      if ( (*(_BYTE *)(v26 + 207) & 0x20) == 0 )
        sub_8B1A30(v25[11], (FILE *)dword_4F07508);
      v22 = (__int64 *)*a3;
      v14 = *(__int64 ***)(v26 + 152);
      v20 = (const __m128i *)*a3;
      if ( v14 )
        goto LABEL_11;
      goto LABEL_15;
    }
  }
  else
  {
    sub_864110((__int64)v12, a2, v8, v9, v10, v11);
  }
  v22 = (__int64 *)*a3;
LABEL_15:
  v14 = (__int64 **)v22;
  if ( v22 )
  {
    v14 = 0;
    sub_725130(v22);
    *a3 = 0;
    return v14;
  }
  v20 = 0;
LABEL_11:
  sub_894B30(v3, v6, v20, 0, (__int64)v14, v19);
  return v14;
}
