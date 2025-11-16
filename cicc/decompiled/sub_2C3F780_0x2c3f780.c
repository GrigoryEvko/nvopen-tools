// Function: sub_2C3F780
// Address: 0x2c3f780
//
__int64 __fastcall sub_2C3F780(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rdx
  __int64 v9[2]; // [rsp+8h] [rbp-38h] BYREF
  _QWORD *v10; // [rsp+18h] [rbp-28h] BYREF

  v9[0] = a2;
  if ( !a3 || !sub_2BF04A0(v9[0]) )
    return v9[0];
  if ( (unsigned __int8)sub_2C3F050(a1 + 160, v9, &v10) )
  {
    v5 = v10 + 1;
  }
  else
  {
    v6 = sub_2C3F6F0(a1 + 160, v9, v10);
    v7 = v9[0];
    v6[2] = 0x600000000LL;
    *v6 = v7;
    v8 = v6 + 3;
    v5 = v6 + 1;
    *v5 = v8;
  }
  return *(_QWORD *)(*v5 + 8LL * (unsigned int)(a3 - 1));
}
