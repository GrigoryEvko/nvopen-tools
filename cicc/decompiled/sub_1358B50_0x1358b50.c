// Function: sub_1358B50
// Address: 0x1358b50
//
__int64 __fastcall sub_1358B50(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  unsigned __int64 v11; // rcx
  __int64 v12; // r9
  unsigned __int8 v13; // al
  unsigned __int64 v14; // r15
  __int64 *v15; // rbx
  __m128i v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // [rsp-78h] [rbp-78h]
  __int64 v20; // [rsp-70h] [rbp-70h]
  __m128i v21; // [rsp-68h] [rbp-68h] BYREF
  __int64 v22; // [rsp-58h] [rbp-58h]
  __int64 v23; // [rsp-50h] [rbp-50h]
  __int64 v24; // [rsp-48h] [rbp-48h]
  char v25; // [rsp-40h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 67) & 8) != 0 )
    return 1;
  if ( !(unsigned __int8)sub_15F2ED0(a2) && !(unsigned __int8)sub_15F3040(a2) )
    return 0;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(a1 + 48) - v6) >> 3);
  if ( (_DWORD)v7 )
  {
    v8 = 0;
    v19 = 24LL * (unsigned int)v7;
    while ( 1 )
    {
      v9 = *(_QWORD *)(v6 + v8 + 16);
      if ( v9 )
      {
        v10 = *(_BYTE *)(v9 + 16);
        if ( v10 <= 0x17u )
        {
          v11 = 0;
          v12 = 0;
        }
        else if ( v10 == 78 )
        {
          v12 = v9 | 4;
          v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        }
        else
        {
          v11 = 0;
          v12 = 0;
          if ( v10 == 29 )
          {
            v12 = v9 & 0xFFFFFFFFFFFFFFFBLL;
            v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          }
        }
        v13 = *(_BYTE *)(a2 + 16);
        if ( v13 <= 0x17u )
          return 1;
        if ( v13 == 78 )
        {
          v14 = a2 & 0xFFFFFFFFFFFFFFFBLL | 4;
        }
        else
        {
          if ( v13 != 29 )
            return 1;
          v14 = a2 & 0xFFFFFFFFFFFFFFFBLL;
        }
        if ( !v11 )
          return 1;
        if ( (v14 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          return 1;
        v20 = v12;
        if ( (sub_134F530(a3, v12, v14) & 3) != 0 || (sub_134F530(a3, v14, v20) & 3) != 0 )
          return 1;
      }
      v8 += 24;
      if ( v19 == v8 )
        break;
      v6 = *(_QWORD *)(a1 + 40);
    }
  }
  v15 = *(__int64 **)(a1 + 16);
  if ( v15 )
  {
    while ( 1 )
    {
      v17 = v15[5];
      v18 = v15[7];
      if ( (v17 == -8 || v17 == -16) && !v15[6] && !v18 )
        v17 = 0;
      v16.m128i_i64[0] = *v15;
      v16.m128i_i64[1] = v15[4];
      v23 = v15[6];
      v25 = 1;
      v21 = v16;
      v22 = v17;
      v24 = v18;
      if ( (sub_13575E0(a3, a2, &v21, v18) & 3) != 0 )
        break;
      v15 = (__int64 *)v15[2];
      if ( !v15 )
        return 0;
    }
    return 1;
  }
  return 0;
}
