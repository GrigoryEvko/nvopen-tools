// Function: sub_8B46F0
// Address: 0x8b46f0
//
__int64 __fastcall sub_8B46F0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  _UNKNOWN *__ptr32 *v11; // r8
  __int64 v12; // r15
  int v13; // eax
  __int64 v14; // rsi
  __int64 result; // rax
  __int64 v16; // rax
  __int64 *v17; // r12
  unsigned int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // rsi
  bool v21; // dl
  int v22; // r8d
  bool v23; // al
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-58h]
  _BOOL4 v26; // [rsp+8h] [rbp-58h]
  int v27; // [rsp+14h] [rbp-4Ch] BYREF
  int v28; // [rsp+18h] [rbp-48h] BYREF
  int v29; // [rsp+1Ch] [rbp-44h] BYREF
  const __m128i *v30; // [rsp+20h] [rbp-40h] BYREF
  __m128i *v31; // [rsp+28h] [rbp-38h] BYREF

  v8 = sub_730E00(a2);
  v12 = sub_730E00(a1);
  if ( *(_BYTE *)(v8 + 173) != 12 )
    goto LABEL_6;
  if ( !*(_BYTE *)(v8 + 176) )
  {
    v13 = 0;
    if ( a4 )
      v13 = *(_DWORD *)(sub_892BC0(a4) + 4);
    if ( *(_DWORD *)(v8 + 188) != v13 )
    {
LABEL_6:
      v14 = v8;
      return sub_73A2C0(v12, v14, v9, v10, v11);
    }
  }
  v16 = sub_730E00(v8);
  if ( *(_BYTE *)(v16 + 173) != 12 || *(_BYTE *)(v16 + 176) )
  {
    if ( *(_BYTE *)(v8 + 176) == 1 && (unsigned int)sub_72E9D0((_BYTE *)v8, &v30, &v27) )
    {
      if ( (unsigned int)sub_8D2780(*(_QWORD *)(v12 + 128))
        && (unsigned int)sub_8B3500(*(__m128i **)(v12 + 128), *(_QWORD *)(v8 + 128), a3, a4, 0) )
      {
        if ( v30[10].m128i_i8[13] != 12 )
        {
          v31 = (__m128i *)sub_724DC0();
          sub_72A510(v30, v31);
          sub_7115B0(v31, *(_QWORD *)(v12 + 128), 1, 1, 1, 1, 0, 0, 1u, 0, 0, &v28, &v29, dword_4F07508);
          if ( v29 )
            v28 = 1;
          v23 = 0;
          if ( !v28 )
            v23 = (unsigned int)sub_8B46F0(v12, v31, a3, a4, a5) != 0;
          v26 = v23;
          sub_724E30((__int64)&v31);
          return v26;
        }
        sub_8B46F0(v12, v30, a3, a4, a5);
      }
      else if ( v30[10].m128i_i8[13] != 12 )
      {
        return sub_8B46F0(v12, v30, a3, a4, a5);
      }
    }
    return 1;
  }
  v25 = v16;
  v17 = sub_8A4360(a4, a3, (unsigned int *)(v16 + 184), 0, 0);
  if ( (v17[3] & 1) != 0 )
  {
    if ( (unsigned int)sub_8D2780(*(_QWORD *)(v12 + 128)) )
    {
      v22 = sub_621100(v12, v17[4]);
      result = 0;
      if ( !v22 )
      {
        *((_BYTE *)v17 + 24) &= ~1u;
        v17[4] = v12;
        return 1;
      }
      return result;
    }
    return 0;
  }
  v18 = *(_DWORD *)(v25 + 184);
  v9 = a4;
  if ( v18 > 1 )
  {
    do
    {
      --v18;
      v9 = *(_QWORD *)v9;
    }
    while ( v18 != 1 );
  }
  if ( (*(_BYTE *)(v9 + 57) & 8) != 0 )
  {
    result = sub_8B3500(*(__m128i **)(v12 + 128), *(_QWORD *)(v25 + 128), a3, a4, 0);
    if ( (_DWORD)result )
      goto LABEL_21;
    return 0;
  }
  if ( (*(_BYTE *)(v9 + 72) & 1) != 0 )
  {
    if ( dword_4D04804 )
      sub_8B3500(*(__m128i **)(v12 + 128), *(_QWORD *)(v25 + 128), a3, a4, 0);
    goto LABEL_20;
  }
  v19 = *(_QWORD *)(v12 + 128);
  v20 = *(_QWORD *)(v25 + 128);
  if ( v19 != v20 && !(unsigned int)sub_8D97D0(v19, v20, 0, v10, v11) )
  {
    if ( !(unsigned int)sub_8DBE70(*(_QWORD *)(v12 + 128))
      || !(unsigned int)sub_8D3EA0(*(_QWORD *)(v12 + 128))
      && (v24 = sub_8D4940(*(_QWORD *)(v12 + 128)), (unsigned int)sub_8D3EA0(v24)) )
    {
      if ( (_DWORD)qword_4F077B4 )
      {
        if ( (a5 & 0x600) != 0x600 )
          return 0;
        goto LABEL_20;
      }
      return 0;
    }
  }
LABEL_20:
  result = 1;
LABEL_21:
  v14 = v17[4];
  if ( v14 )
    return sub_73A2C0(v12, v14, v9, v10, v11);
  v17[4] = v12;
  v21 = 0;
  if ( *(_BYTE *)(v12 + 173) == 12 )
    v21 = (*(_BYTE *)(v12 + 177) & 4) != 0;
  *((_BYTE *)v17 + 24) = (16 * v21) | v17[3] & 0xEF;
  return result;
}
