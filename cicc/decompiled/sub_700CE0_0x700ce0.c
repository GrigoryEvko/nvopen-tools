// Function: sub_700CE0
// Address: 0x700ce0
//
__int64 __fastcall sub_700CE0(__m128i *a1, __m128i *a2, __m128i *a3, __int64 a4, __int64 *a5, _QWORD *a6, _QWORD *a7)
{
  unsigned int v9; // ebx
  __int8 v10; // al
  __int8 v11; // al
  __int8 v12; // al
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9

  v9 = a4;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
  {
    sub_6F40C0((__int64)a1, (__int64)a2, (__int64)a3, a4, (__int64)a5, (__int64)a6);
    sub_6F40C0((__int64)a2, (__int64)a2, v13, v14, v15, v16);
    sub_6F40C0((__int64)a3, (__int64)a2, v17, v18, v19, v20);
    return sub_6FFD30(a1, a2, a3, dword_4D03B80, 0, 1, 0, 1, v9, a5, a6, a7);
  }
  v10 = a1[1].m128i_i8[0];
  if ( v10 == 3 )
  {
    sub_6F3BA0(a1, 1);
    v11 = a2[1].m128i_i8[0];
    if ( v11 != 3 )
      goto LABEL_6;
  }
  else
  {
    if ( v10 == 4 )
      sub_6EE880((__int64)a1, 0);
    sub_6F69D0(a1, 0);
    v11 = a2[1].m128i_i8[0];
    if ( v11 != 3 )
    {
LABEL_6:
      if ( v11 == 4 )
        sub_6EE880((__int64)a2, 0);
      sub_6F69D0(a2, 0);
      v12 = a3[1].m128i_i8[0];
      if ( v12 != 3 )
        goto LABEL_9;
LABEL_16:
      sub_6F3BA0(a3, 1);
      return sub_6FFD30(a1, a2, a3, dword_4D03B80, 0, 1, 0, 1, v9, a5, a6, a7);
    }
  }
  sub_6F3BA0(a2, 1);
  v12 = a3[1].m128i_i8[0];
  if ( v12 == 3 )
    goto LABEL_16;
LABEL_9:
  if ( v12 == 4 )
    sub_6EE880((__int64)a3, 0);
  sub_6F69D0(a3, 0);
  return sub_6FFD30(a1, a2, a3, dword_4D03B80, 0, 1, 0, 1, v9, a5, a6, a7);
}
