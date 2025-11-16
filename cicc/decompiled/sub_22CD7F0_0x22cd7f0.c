// Function: sub_22CD7F0
// Address: 0x22cd7f0
//
__int64 __fastcall sub_22CD7F0(__int64 a1, unsigned __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  unsigned __int8 v5; // al
  __int64 v7; // r15
  unsigned __int64 v8; // rax
  int v9; // edx
  int v10; // eax
  unsigned __int8 v11; // dl
  unsigned __int8 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-88h]
  __m128i v15; // [rsp+10h] [rbp-80h] BYREF
  __int64 v16; // [rsp+20h] [rbp-70h]
  __int64 v17; // [rsp+28h] [rbp-68h]
  __int64 v18; // [rsp+30h] [rbp-60h]
  __int64 v19; // [rsp+38h] [rbp-58h]
  __int64 v20; // [rsp+40h] [rbp-50h]
  __int64 v21; // [rsp+48h] [rbp-48h]
  __int16 v22; // [rsp+50h] [rbp-40h]

  v5 = *a3;
  if ( *a3 > 0x1Cu && a4 == *((_QWORD *)a3 + 5) )
  {
    if ( v5 == 84 )
    {
      sub_22CD5E0(a1, a2, (__int64)a3, a4);
    }
    else
    {
      if ( v5 != 86 )
      {
        v7 = *((_QWORD *)a3 + 1);
        if ( *(_BYTE *)(v7 + 8) == 14 )
        {
          v8 = *(_QWORD *)(a2 + 248);
          v22 = 257;
          v14 = a4;
          v15 = (__m128i)v8;
          v16 = 0;
          v17 = 0;
          v18 = 0;
          v19 = 0;
          v20 = 0;
          v21 = 0;
          if ( (unsigned __int8)sub_9B6260((__int64)a3, &v15, 0) )
          {
            v12 = (unsigned __int8 *)sub_AC9EC0((__int64 **)v7);
            v15.m128i_i16[0] = 0;
            sub_22C0430((__int64)&v15, v12);
            sub_22C0650(a1, (unsigned __int8 *)&v15);
            *(_BYTE *)(a1 + 40) = 1;
            sub_22C0090((unsigned __int8 *)&v15);
            return a1;
          }
          v7 = *((_QWORD *)a3 + 1);
          a4 = v14;
        }
        v9 = *(unsigned __int8 *)(v7 + 8);
        if ( (unsigned int)(v9 - 17) <= 1 )
          LOBYTE(v9) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
        if ( (_BYTE)v9 == 12 )
        {
          v10 = *a3;
          v11 = *a3;
          if ( (unsigned int)(v10 - 67) <= 0xC )
          {
            sub_22C7910(a1, a2, a3, a4);
            return a1;
          }
          if ( (unsigned int)(v10 - 42) <= 0x11 )
          {
            sub_22CB500(a1, a2, a3, a4);
            return a1;
          }
          switch ( v11 )
          {
            case '[':
              sub_22C7F20(a1, a2, (__int64)a3, a4);
              return a1;
            case ']':
              sub_22CCC80(a1, a2, (__int64)a3, a4);
              return a1;
            case 'U':
              v13 = *((_QWORD *)a3 - 4);
              if ( v13 )
              {
                if ( !*(_BYTE *)v13
                  && *(_QWORD *)(v13 + 24) == *((_QWORD *)a3 + 10)
                  && (*(_BYTE *)(v13 + 33) & 0x20) != 0 )
                {
                  sub_22C7A70(a1, a2, a3, a4);
                  return a1;
                }
              }
              break;
          }
        }
        sub_22C07D0(&v15, a3);
        sub_22C0650(a1, (unsigned __int8 *)&v15);
        *(_BYTE *)(a1 + 40) = 1;
        sub_22C0090((unsigned __int8 *)&v15);
        return a1;
      }
      sub_22CC150(a1, a2, (__int64)a3, a4);
    }
  }
  else
  {
    sub_22CD2E0(a1, a2, (__int64)a3, a4);
  }
  return a1;
}
