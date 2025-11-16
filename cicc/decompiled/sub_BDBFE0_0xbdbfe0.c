// Function: sub_BDBFE0
// Address: 0xbdbfe0
//
void __fastcall sub_BDBFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  const char *v9; // rax
  __int64 v10; // r12
  _BYTE *v11; // rax
  __m128i v12[2]; // [rsp+0h] [rbp-B0h] BYREF
  char v13; // [rsp+20h] [rbp-90h]
  char v14; // [rsp+21h] [rbp-8Fh]
  __m128i v15; // [rsp+30h] [rbp-80h] BYREF
  __int16 v16; // [rsp+50h] [rbp-60h]
  __m128i v17; // [rsp+60h] [rbp-50h] BYREF
  __int64 v18; // [rsp+70h] [rbp-40h]
  __int64 v19; // [rsp+78h] [rbp-38h]
  __int16 v20; // [rsp+80h] [rbp-30h]

  if ( sub_A75040(a2, 83) )
  {
    v9 = "inalloca attribute not allowed in ";
  }
  else if ( sub_A75040(a2, 15) )
  {
    v9 = "inreg attribute not allowed in ";
  }
  else
  {
    if ( !sub_A75040(a2, 74) )
    {
      if ( sub_A75040(a2, 84) )
      {
        v18 = a3;
        v19 = a4;
        v17.m128i_i64[0] = (__int64)"preallocated attribute not allowed in ";
        v20 = 1283;
        sub_BDBF70((__int64 *)a1, (__int64)&v17);
      }
      else if ( sub_A75040(a2, 80) )
      {
        v15.m128i_i64[0] = a3;
        v16 = 261;
        v15.m128i_i64[1] = a4;
        v14 = 1;
        v12[0].m128i_i64[0] = (__int64)"byref attribute not allowed in ";
        v13 = 3;
        sub_9C6370(&v17, v12, &v15, v6, v7, v8);
        sub_BDBF70((__int64 *)a1, (__int64)&v17);
      }
      return;
    }
    v9 = "swifterror attribute not allowed in ";
  }
  v10 = *(_QWORD *)a1;
  v17.m128i_i64[0] = (__int64)v9;
  v18 = a3;
  v19 = a4;
  v20 = 1283;
  if ( v10 )
  {
    sub_CA0E80(&v17, v10);
    v11 = *(_BYTE **)(v10 + 32);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
    {
      sub_CB5D20(v10, 10);
    }
    else
    {
      *(_QWORD *)(v10 + 32) = v11 + 1;
      *v11 = 10;
    }
  }
  *(_BYTE *)(a1 + 152) = 1;
}
