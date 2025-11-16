// Function: sub_6368A0
// Address: 0x6368a0
//
__int64 __fastcall sub_6368A0(__int64 **a1, __int64 a2, __m128i *a3, __int64 *a4)
{
  __int64 *v7; // rax
  char v8; // bl
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 result; // rax
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v16[5]; // [rsp+18h] [rbp-28h] BYREF

  v7 = *a1;
  v8 = (unsigned __int8)a3[2].m128i_i8[9] >> 7;
  v15 = a2;
  *a4 = 0;
  v16[0] = v7;
  a3[2].m128i_i8[9] |= 0x80u;
  if ( (unsigned int)sub_8D3A70(a2) )
  {
    a3[2].m128i_i8[9] |= 0x40u;
    v9 = sub_6E1A20(v16[0]);
    sub_6333F0((__int64 *)v16, v15, a3, v9, a4);
    v10 = v16[0];
  }
  else if ( (unsigned int)sub_8D3410(v15) )
  {
    a3[2].m128i_i8[9] |= 0x40u;
    v12 = (_QWORD *)sub_6E1A20(v16[0]);
    sub_635980(v16, &v15, a3, v12, a4);
    v10 = v16[0];
  }
  else
  {
    a3[2].m128i_i8[9] |= 2u;
    if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
    {
      if ( (unsigned int)sub_8D3D40(v15) )
      {
        v13 = sub_6E1A20(v16[0]);
        sub_6851C0(1797, v13);
      }
      else
      {
        v14 = sub_6E1A20(v16[0]);
        sub_685360(2357, v14);
      }
    }
    v10 = v16[0];
    while ( v10 && *((_BYTE *)v10 + 8) == 2 )
    {
      if ( !*v10 )
      {
        v10 = 0;
        break;
      }
      if ( *(_BYTE *)(*v10 + 8) == 3 )
        v10 = (__int64 *)sub_6BBB10(v10);
      else
        v10 = (__int64 *)*v10;
    }
  }
  *a1 = v10;
  result = a3[2].m128i_i8[9] & 0x7F;
  a3[2].m128i_i8[9] = result | (v8 << 7);
  return result;
}
