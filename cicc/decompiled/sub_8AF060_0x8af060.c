// Function: sub_8AF060
// Address: 0x8af060
//
__int64 *__fastcall sub_8AF060(__int64 a1, __m128i **a2)
{
  __int64 *v3; // rax
  _QWORD *v4; // r14
  __int64 *v5; // r15
  __int64 v6; // r13
  int v7; // ebx
  int v8; // r8d
  int v9; // ecx
  char v10; // di
  int v11; // esi
  __int64 *result; // rax
  __int64 *v13; // rax
  unsigned __int8 v14; // di
  char v15; // al
  _QWORD *v16; // rax
  int v17; // [rsp+4h] [rbp-3Ch]

  v3 = *(__int64 **)(*(_QWORD *)(a1 + 88) + 256LL);
  v4 = (_QWORD *)*v3;
  if ( *v3 )
  {
    v5 = (__int64 *)a2;
    v6 = *v3;
    v7 = 0;
    v8 = 0;
    do
    {
      while ( 1 )
      {
        result = (__int64 *)*v5;
        if ( *v5 )
          break;
        if ( (*(_BYTE *)(v6 + 56) & 0x10) != 0 )
        {
          if ( v8 )
            return sub_8A0370(a1, a2, 0, 0, 0, 0, 0);
LABEL_14:
          v13 = sub_725090(3u);
          *v13 = *v5;
          *v5 = (__int64)v13;
          v5 = v13;
          result = (__int64 *)*v13;
          if ( !result )
            return sub_8A0370(a1, a2, 0, 0, 0, 0, 0);
          goto LABEL_15;
        }
        if ( v7 )
          return sub_8A0370(a1, a2, 0, 0, 0, 0, 0);
        if ( (*(_BYTE *)(v6 + 56) & 1) == 0 )
          return result;
        v14 = 0;
        v15 = *(_BYTE *)(*(_QWORD *)(v6 + 8) + 80LL);
        if ( v15 != 3 )
          v14 = (v15 != 2) + 1;
        v17 = v8;
        v16 = sub_725090(v14);
        *v5 = (__int64)v16;
        sub_8AEEA0(a1, (__int64)v16, v6, v4);
        result = (__int64 *)*v5;
        v6 = *(_QWORD *)v6;
        v8 = v17;
        if ( !*v5 )
          return sub_8A0370(a1, a2, 0, 0, 0, 0, 0);
LABEL_9:
        if ( v8 )
          goto LABEL_15;
        v5 = (__int64 *)*v5;
        if ( !v6 )
          return sub_8A0370(a1, a2, 0, 0, 0, 0, 0);
      }
      v9 = *((unsigned __int8 *)result + 8);
      if ( (_BYTE)v9 == 3 )
      {
        v7 = 1;
        if ( (*(_BYTE *)(v6 + 56) & 0x10) == 0 )
          goto LABEL_8;
      }
      else
      {
        v10 = *(_BYTE *)(*(_QWORD *)(v6 + 8) + 80LL);
        v11 = 0;
        if ( v10 != 3 )
          v11 = (v10 != 2) + 1;
        if ( v9 != v11 )
          return 0;
        if ( (*(_BYTE *)(v6 + 56) & 0x10) == 0 )
        {
LABEL_8:
          v6 = *(_QWORD *)v6;
          goto LABEL_9;
        }
      }
      if ( !v8 )
        goto LABEL_14;
LABEL_15:
      *((_BYTE *)result + 24) |= 8u;
      v8 = 1;
      v5 = (__int64 *)*v5;
    }
    while ( v6 );
  }
  return sub_8A0370(a1, a2, 0, 0, 0, 0, 0);
}
