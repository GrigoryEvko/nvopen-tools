// Function: sub_7A2E10
// Address: 0x7a2e10
//
__int64 __fastcall sub_7A2E10(__int64 a1, __m128i *a2)
{
  _QWORD *v2; // rbx
  __int64 result; // rax
  int v4; // eax
  unsigned int v5; // edx
  unsigned int v6; // r13d
  size_t v7; // rdx
  unsigned int v8; // r13d
  int v9; // ecx
  unsigned int v10; // eax
  bool v11; // zf
  char *v12; // rcx
  char *v13; // rdx
  unsigned int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rcx
  size_t v17; // [rsp+0h] [rbp-110h]
  size_t v18; // [rsp+0h] [rbp-110h]
  __int64 v19; // [rsp+8h] [rbp-108h]
  _BOOL4 i; // [rsp+1Ch] [rbp-F4h] BYREF
  _BYTE v21[16]; // [rsp+20h] [rbp-F0h] BYREF
  void *s; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v23; // [rsp+38h] [rbp-D8h]
  __int64 v24; // [rsp+40h] [rbp-D0h]
  int v25; // [rsp+48h] [rbp-C8h]
  __m128i v26; // [rsp+80h] [rbp-90h] BYREF
  __int64 v27; // [rsp+90h] [rbp-80h]
  char v28; // [rsp+A4h] [rbp-6Ch]

  v2 = *(_QWORD **)a1;
  for ( i = 1; *((_BYTE *)v2 + 140) == 12; v2 = (_QWORD *)v2[20] )
    ;
  result = 1;
  if ( *(_BYTE *)(a1 + 24) != 2 )
  {
    result = dword_4F07588;
    if ( dword_4F07588 )
    {
      result = 0;
      if ( !dword_4D03F94 )
      {
        if ( dword_4F08058 )
        {
          sub_771BE0(a1, dword_4D03F94);
          dword_4F08058 = 0;
        }
        sub_774A30((__int64)v21, 0);
        v27 = *(_QWORD *)(a1 + 28);
        v4 = 32;
        if ( (*(_BYTE *)(a1 + 25) & 3) == 0 )
        {
          v4 = 16;
          if ( (unsigned __int8)(*((_BYTE *)v2 + 140) - 2) > 1u )
            v4 = sub_7764B0((__int64)v21, (unsigned __int64)v2, &i);
        }
        if ( i )
        {
          if ( (unsigned __int8)(*((_BYTE *)v2 + 140) - 8) > 3u )
          {
            v19 = 16;
            v7 = 8;
            v6 = 16;
          }
          else
          {
            v5 = (unsigned int)(v4 + 7) >> 3;
            v6 = v5 + 9;
            if ( (((_BYTE)v5 + 9) & 7) != 0 )
              v6 = v5 + 17 - (((_BYTE)v5 + 9) & 7);
            v19 = v6;
            v7 = v6 - 8LL;
          }
          v8 = v4 + v6;
          if ( v8 > 0x400 )
          {
            v17 = v7;
            v14 = v8 + 16;
            v15 = sub_822B10(v14);
            v16 = v24;
            v7 = v17;
            *(_DWORD *)(v15 + 8) = v14;
            *(_QWORD *)v15 = v16;
            *(_DWORD *)(v15 + 12) = v25;
            v12 = (char *)(v15 + 16);
            v24 = v15;
          }
          else
          {
            v9 = v8 & 7;
            v10 = v8 + 8 - v9;
            v11 = v9 == 0;
            v12 = (char *)s;
            if ( !v11 )
              v8 = v10;
            if ( 0x10000 - ((int)s - (int)v23) < v8 )
            {
              v18 = v7;
              sub_772E70(&s);
              v12 = (char *)s;
              v7 = v18;
            }
            s = &v12[v8];
          }
          v13 = (char *)memset(v12, 0, v7) + v19;
          *((_QWORD *)v13 - 1) = v2;
          if ( (unsigned __int8)(*((_BYTE *)v2 + 140) - 9) <= 2u )
            *(_QWORD *)v13 = 0;
          if ( !(unsigned int)sub_786210((__int64)v21, (_QWORD **)a1, (unsigned __int64)v13, v13) && (v28 & 0x40) == 0 )
            i = 0;
        }
        else
        {
          i = (v28 & 0x40) != 0;
        }
        *a2 = _mm_loadu_si128(&v26);
        sub_771990((__int64)v21);
        return i;
      }
    }
  }
  return result;
}
