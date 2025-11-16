// Function: sub_845C60
// Address: 0x845c60
//
__int64 __fastcall sub_845C60(__int64 a1, __m128i *a2, unsigned int a3, int a4, _DWORD *a5)
{
  __int64 result; // rax
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __m128i *v11; // rax
  __m128i *v12; // r14
  const __m128i *i; // r13
  __m128i *v14; // rbx
  _QWORD *v15; // r13
  unsigned int v16; // [rsp+18h] [rbp-78h] BYREF
  unsigned int v17; // [rsp+1Ch] [rbp-74h] BYREF
  const __m128i *v18; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v19; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-60h] BYREF
  char v21; // [rsp+40h] [rbp-50h]

  result = sub_8D3A70(*(_QWORD *)a1);
  if ( (_DWORD)result )
  {
    if ( (unsigned int)sub_840360(
                         (__int64 *)a1,
                         (__int64)a2,
                         a3,
                         0,
                         1,
                         (a4 & 0x800) == 0,
                         0,
                         0,
                         a4,
                         (__int64)v20,
                         &v16,
                         &v18) )
    {
      v21 &= ~4u;
      if ( (a4 & 0x80000) != 0 && a2 && v20[0] )
      {
        v9 = sub_724DC0();
        v10 = v20[0];
        v19 = v9;
        v11 = sub_73D790(*(_QWORD *)(v20[0] + 152LL));
        v17 = 0;
        v12 = v11;
        if ( !(unsigned int)sub_6E47F0(v10, a1, (__int64)v11, (__int64)v19)
          || !(unsigned int)sub_8DD4B0(v12, 1, v19, a2, &v17) )
        {
          if ( v17 )
            sub_685750(7u, v17, (_DWORD *)(a1 + 68), *(_QWORD *)a1, (__int64)a2);
        }
        sub_724E30((__int64)&v19);
      }
      result = sub_8449E0((_QWORD *)a1, a2, (__int64)v20, 0, 0);
      *a5 = 1;
    }
    else
    {
      result = v16;
      if ( v16 )
      {
        if ( v18 )
        {
          if ( (unsigned int)sub_6E5430() )
          {
            v15 = sub_67DA80(0x1A2u, (_DWORD *)(a1 + 68), *(_QWORD *)a1);
            sub_82E650(v18->m128i_i64, 0, 0, 0, v15);
            sub_685910((__int64)v15, 0);
          }
          for ( i = v18; i; qword_4D03C68 = v14->m128i_i64 )
          {
            v14 = (__m128i *)i;
            i = (const __m128i *)i->m128i_i64[0];
            sub_725130((__int64 *)v14[2].m128i_i64[1]);
            sub_82D8A0((_QWORD *)v14[7].m128i_i64[1]);
            v14->m128i_i64[0] = (__int64)qword_4D03C68;
          }
        }
        result = sub_6E6840(a1);
        *a5 = 1;
      }
    }
  }
  return result;
}
