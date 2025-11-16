// Function: sub_1C54F70
// Address: 0x1c54f70
//
void __fastcall sub_1C54F70(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  bool v7; // al
  __int64 v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rax
  __int64 *v11; // rax
  _QWORD *v12; // [rsp+0h] [rbp-80h] BYREF
  __int64 v13; // [rsp+8h] [rbp-78h]
  _BYTE v14[112]; // [rsp+10h] [rbp-70h] BYREF

  v7 = sub_14560B0(a2);
  a1[1] = a2;
  if ( v7 )
  {
    v10 = sub_1456040(a2);
    *a1 = sub_145CF80(a3, v10, 0, 0);
    return;
  }
  v12 = v14;
  v13 = 0x800000000LL;
  v8 = sub_1456040(a2);
  *a1 = sub_145CF80(a3, v8, 0, 0);
  sub_1C54710(a2, 0, (__int64)&v12, a3, a1, a4, a5);
  if ( !(_DWORD)v13 )
  {
    v9 = v12;
    a1[1] = 0;
    if ( v9 == v14 )
      return;
LABEL_10:
    _libc_free((unsigned __int64)v9);
    return;
  }
  if ( (unsigned int)v13 == 1 )
  {
    v9 = v12;
    a1[1] = *v12;
  }
  else
  {
    v11 = sub_147DD40(a3, (__int64 *)&v12, 0, 0, a4, a5);
    v9 = v12;
    a1[1] = (__int64)v11;
  }
  if ( v9 != v14 )
    goto LABEL_10;
}
