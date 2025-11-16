// Function: sub_278B9D0
// Address: 0x278b9d0
//
__int64 __fastcall sub_278B9D0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v8; // r10
  unsigned int v9; // ecx
  int *v10; // rax
  int v11; // edi
  _DWORD *v12; // rax
  _BYTE *v13; // r12
  unsigned int v14; // eax
  unsigned __int64 v15; // rax
  const __m128i **v17; // rdx
  const __m128i *v18; // rax
  const __m128i *v19; // rsi
  int v20; // eax
  int v21; // r11d

  v6 = *(unsigned int *)(a6 + 376);
  v8 = *(_QWORD *)(a6 + 360);
  if ( (_DWORD)v6 )
  {
    v9 = (v6 - 1) & (37 * a2);
    v10 = (int *)(v8 + 40LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_3:
      if ( v10 != (int *)(v8 + 40 * v6) )
      {
        v12 = v10 + 2;
        while ( 1 )
        {
          v13 = *(_BYTE **)v12;
          if ( **(_BYTE **)v12 == 85 )
          {
            if ( *((_QWORD *)v13 + 5) == a5 )
              goto LABEL_9;
          }
          else
          {
            v13 = 0;
          }
          v12 = (_DWORD *)*((_QWORD *)v12 + 3);
          if ( !v12 )
            goto LABEL_9;
        }
      }
    }
    else
    {
      v20 = 1;
      while ( v11 != -1 )
      {
        v21 = v20 + 1;
        v9 = (v6 - 1) & (v20 + v9);
        v10 = (int *)(v8 + 40LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_3;
        v20 = v21;
      }
    }
  }
  v13 = 0;
LABEL_9:
  if ( (unsigned int)sub_CF5CA0(*(_QWORD *)(a1 + 184), (__int64)v13) )
  {
    if ( !*(_QWORD *)(a1 + 192) )
      return 0;
    v14 = sub_CF5CA0(*(_QWORD *)(a1 + 184), (__int64)v13);
    if ( (((unsigned __int8)((v14 >> 4) | v14 | (v14 >> 2)) | (unsigned __int8)(v14 >> 6)) & 2) != 0 )
      return 0;
    v15 = sub_1037A30(*(_QWORD *)(a1 + 192), v13, 1);
    if ( (v15 & 7) != 3 )
      return 0;
    if ( v15 >> 61 != 1 )
      return 0;
    v17 = sub_10305C0(*(_QWORD *)(a1 + 192), (__int64)v13);
    v18 = *v17;
    v19 = v17[1];
    if ( v19 == *v17 )
      return 0;
    while ( (v18->m128i_i32[2] & 7) != 3 || (unsigned __int64)v18->m128i_i64[1] >> 61 != 2 )
    {
      if ( v19 == ++v18 )
        return 0;
    }
  }
  return 1;
}
