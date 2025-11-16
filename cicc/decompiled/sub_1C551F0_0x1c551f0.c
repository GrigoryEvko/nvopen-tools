// Function: sub_1C551F0
// Address: 0x1c551f0
//
__int64 __fastcall sub_1C551F0(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // rax
  __int64 result; // rax
  __int64 *v10; // rax
  __int64 v11; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v12; // [rsp+8h] [rbp-98h]
  _BYTE *v13; // [rsp+20h] [rbp-80h] BYREF
  __int64 v14; // [rsp+28h] [rbp-78h]
  _BYTE v15[112]; // [rsp+30h] [rbp-70h] BYREF

  v12 = a1;
  if ( sub_14560B0(a1) )
  {
    v8 = sub_1456040(a1);
    v11 = sub_145CF80(a3, v8, 0, 0);
    goto LABEL_8;
  }
  v13 = v15;
  v14 = 0x800000000LL;
  v6 = sub_1456040(a1);
  v11 = sub_145CF80(a3, v6, 0, 0);
  sub_1C54710(a1, 0, (__int64)&v13, a3, &v11, a4, a5);
  if ( !(_DWORD)v14 )
  {
    v12 = 0;
    v7 = v13;
    if ( v13 == v15 )
      goto LABEL_8;
    goto LABEL_12;
  }
  if ( (unsigned int)v14 == 1 )
  {
    v7 = v13;
    v12 = *(_QWORD *)v13;
  }
  else
  {
    v10 = sub_147DD40(a3, (__int64 *)&v13, 0, 0, a4, a5);
    v7 = v13;
    v12 = (__int64)v10;
  }
  if ( v7 != v15 )
LABEL_12:
    _libc_free((unsigned __int64)v7);
LABEL_8:
  sub_1C54F70((__int64 *)&v13, a2, a3, a4, a5);
  result = 0;
  if ( v12 == v14 )
    return sub_14806B0(a3, v11, (__int64)v13, 0, 0);
  return result;
}
