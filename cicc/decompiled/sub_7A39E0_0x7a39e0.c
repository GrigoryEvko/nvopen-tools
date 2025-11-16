// Function: sub_7A39E0
// Address: 0x7a39e0
//
__int64 __fastcall sub_7A39E0(__int64 a1, int a2, __int64 a3, __m128i *a4)
{
  unsigned __int64 v4; // rbx
  __int64 result; // rax
  int v6; // ecx
  unsigned int v7; // edx
  unsigned int v8; // eax
  __int64 v9; // r15
  size_t v10; // rdx
  __int64 v11; // rax
  char *v12; // rcx
  __m128i *v13; // r15
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rcx
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  __int64 j; // rax
  __int64 v23; // rdx
  size_t v24; // [rsp+0h] [rbp-120h]
  size_t v25; // [rsp+0h] [rbp-120h]
  int v26; // [rsp+Ch] [rbp-114h]
  unsigned int v27; // [rsp+Ch] [rbp-114h]
  unsigned int i; // [rsp+2Ch] [rbp-F4h] BYREF
  _BYTE v31[16]; // [rsp+30h] [rbp-F0h] BYREF
  void *s; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v33; // [rsp+48h] [rbp-D8h]
  __int64 v34; // [rsp+50h] [rbp-D0h]
  int v35; // [rsp+58h] [rbp-C8h]
  _QWORD *v36; // [rsp+60h] [rbp-C0h]
  __m128i v37; // [rsp+90h] [rbp-90h] BYREF
  __int64 v38; // [rsp+A0h] [rbp-80h]
  char v39; // [rsp+B4h] [rbp-6Ch]
  char v40; // [rsp+B5h] [rbp-6Bh]
  int v41; // [rsp+B8h] [rbp-68h]
  __int64 v42; // [rsp+E8h] [rbp-38h]

  v4 = *(_QWORD *)a1;
  for ( i = 1; *(_BYTE *)(v4 + 140) == 12; v4 = *(_QWORD *)(v4 + 160) )
    ;
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
      sub_774A30((__int64)v31, a2);
      if ( a2 )
        v40 |= 0x10u;
      v6 = 32;
      v38 = *(_QWORD *)(a1 + 28);
      if ( (*(_BYTE *)(a1 + 25) & 3) == 0 )
      {
        v6 = 16;
        if ( (unsigned __int8)(*(_BYTE *)(v4 + 140) - 2) > 1u )
          v6 = sub_7764B0((__int64)v31, v4, &i);
      }
      if ( !i )
      {
        if ( (v39 & 0x40) != 0 )
        {
          sub_72C970(a3);
          i = 1;
        }
        goto LABEL_15;
      }
      if ( (unsigned __int8)(*(_BYTE *)(v4 + 140) - 8) > 3u )
      {
        v10 = 8;
        v9 = 16;
        v8 = 16;
      }
      else
      {
        v7 = (unsigned int)(v6 + 7) >> 3;
        v8 = v7 + 9;
        if ( (((_BYTE)v7 + 9) & 7) != 0 )
          v8 = v7 + 17 - (((_BYTE)v7 + 9) & 7);
        v9 = v8;
        v10 = v8 - 8LL;
      }
      v11 = v6 + v8;
      if ( (unsigned int)v11 > 0x400 )
      {
        v24 = v10;
        v26 = v11 + 16;
        v17 = sub_822B10((unsigned int)(v11 + 16));
        v10 = v24;
        *(_QWORD *)v17 = v34;
        *(_DWORD *)(v17 + 8) = v26;
        *(_DWORD *)(v17 + 12) = v35;
        v12 = (char *)(v17 + 16);
        v34 = v17;
      }
      else
      {
        if ( (v11 & 7) != 0 )
          v11 = (_DWORD)v11 + 8 - (unsigned int)(v11 & 7);
        v12 = (char *)s;
        if ( 0x10000 - ((int)s - (int)v33) < (unsigned int)v11 )
        {
          v25 = v10;
          v27 = v11;
          sub_772E70(&s);
          v12 = (char *)s;
          v10 = v25;
          v11 = v27;
        }
        s = &v12[v11];
      }
      v13 = (__m128i *)((char *)memset(v12, 0, v10) + v9);
      v13[-1].m128i_i64[1] = v4;
      if ( (unsigned __int8)(*(_BYTE *)(v4 + 140) - 9) <= 2u )
        v13->m128i_i64[0] = 0;
      if ( !sub_77FCB0((__int64)v31, a1, v13, v13->m128i_i8) )
      {
        if ( (v39 & 0x40) == 0 )
        {
          i = 0;
          v14 = qword_4D03C50;
          if ( qword_4D03C50 )
          {
            v15 = v41;
            if ( (v41 & 1) != 0 )
              *(_BYTE *)(qword_4D03C50 + 24LL) |= 1u;
            if ( (v15 & 2) != 0 )
              *(_BYTE *)(v14 + 24) |= 2u;
          }
          goto LABEL_15;
        }
        sub_72C970(a3);
LABEL_48:
        if ( i )
        {
          if ( a2 )
          {
            v18 = v36;
            if ( v36 )
            {
              v19 = *(_QWORD *)(qword_4D03C50 + 48LL);
              if ( v19 )
              {
                v20 = (_QWORD *)(v19 + 24);
                v21 = v36[1];
                for ( j = *(_QWORD *)(v19 + 24); ; j = v23 )
                {
                  v23 = *(_QWORD *)(j + 32);
                  if ( j == v21 )
                  {
                    *v20 = v23;
                    v18 = (_QWORD *)*v18;
                    if ( !v18 )
                      goto LABEL_15;
                    v21 = v18[1];
                  }
                  else
                  {
                    v20 = (_QWORD *)(j + 32);
                  }
                }
              }
            }
          }
        }
        goto LABEL_15;
      }
      if ( sub_77D750((__int64)v31, v13, (__int64)v13, *(_QWORD *)a1, a3) )
      {
        if ( a2 )
        {
          if ( v36 && !(unsigned int)sub_799890((__int64)v31) )
            goto LABEL_45;
        }
        else
        {
          if ( v36 )
            goto LABEL_45;
          if ( (unsigned __int8)(*(_BYTE *)(v4 + 140) - 9) <= 2u )
          {
            v16 = *(_QWORD *)(*(_QWORD *)v4 + 96LL);
            if ( *(_QWORD *)(v16 + 24) )
            {
              if ( (*(_BYTE *)(v16 + 177) & 2) == 0 )
                goto LABEL_45;
            }
          }
        }
        if ( !v42 )
        {
          if ( !*(_QWORD *)(a1 + 16) && (!qword_4D03C50 || *(_BYTE *)(qword_4D03C50 + 16LL)) )
            *(_QWORD *)(a3 + 144) = a1;
          goto LABEL_48;
        }
        sub_773640((__int64)v31);
      }
LABEL_45:
      i = 0;
LABEL_15:
      *a4 = _mm_loadu_si128(&v37);
      sub_771990((__int64)v31);
      return i;
    }
  }
  return result;
}
