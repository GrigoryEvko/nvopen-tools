// Function: sub_1C55090
// Address: 0x1c55090
//
__int64 __fastcall sub_1C55090(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v15; // [rsp+18h] [rbp-98h]
  _BYTE *v16; // [rsp+30h] [rbp-80h] BYREF
  __int64 v17; // [rsp+38h] [rbp-78h]
  _BYTE v18[112]; // [rsp+40h] [rbp-70h] BYREF

  v15 = a1;
  if ( !sub_14560B0(a1) )
  {
    v16 = v18;
    v17 = 0x800000000LL;
    v8 = sub_1456040(a1);
    v14 = sub_145CF80(a2, v8, 0, 0);
    sub_1C54710(a1, 0, (__int64)&v16, a2, &v14, a5, a6);
    if ( (_DWORD)v17 )
    {
      if ( (unsigned int)v17 == 1 )
      {
        v9 = v16;
        v15 = *(_QWORD *)v16;
      }
      else
      {
        v13 = sub_147DD40(a2, (__int64 *)&v16, 0, 0, a5, a6);
        v9 = v16;
        v15 = (__int64)v13;
      }
      if ( v9 == v18 )
        goto LABEL_11;
    }
    else
    {
      v9 = v16;
      v12 = 0;
      v15 = 0;
      if ( v16 == v18 )
      {
LABEL_12:
        result = v14;
        goto LABEL_8;
      }
    }
    _libc_free((unsigned __int64)v9);
LABEL_11:
    v12 = v15;
    goto LABEL_12;
  }
  v10 = sub_1456040(a1);
  result = sub_145CF80(a2, v10, 0, 0);
  v12 = v15;
LABEL_8:
  *a3 = v12;
  *a4 = result;
  return result;
}
