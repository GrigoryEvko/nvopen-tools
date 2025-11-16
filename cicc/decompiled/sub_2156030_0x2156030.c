// Function: sub_2156030
// Address: 0x2156030
//
__int64 __fastcall sub_2156030(__int64 a1, _BYTE *a2, __int64 a3, unsigned int a4, char a5)
{
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // [rsp+8h] [rbp-88h]
  __int64 *v12; // [rsp+10h] [rbp-80h] BYREF
  __int16 v13; // [rsp+20h] [rbp-70h]
  __int64 v14[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v15[2]; // [rsp+40h] [rbp-50h] BYREF
  char v16; // [rsp+50h] [rbp-40h]

  if ( a2 )
  {
    v14[0] = (__int64)v15;
    sub_214AFD0(v14, a2, (__int64)&a2[a3]);
  }
  else
  {
    v14[1] = 0;
    v14[0] = (__int64)v15;
    LOBYTE(v15[0]) = 0;
  }
  v7 = sub_21558D0(a1, (__int64)v14);
  v8 = v7;
  if ( (_QWORD *)v14[0] != v15 )
  {
    v11 = v7;
    j_j___libc_free_0(v14[0], v15[0] + 1LL);
    v8 = v11;
  }
  result = sub_2155AB0((__int64)v14, v8, a4, a5);
  if ( v16 )
  {
    v10 = *(_QWORD *)(a1 + 256);
    v12 = v14;
    v13 = 260;
    result = sub_38DD5A0(v10, &v12);
    if ( v16 )
    {
      if ( (_QWORD *)v14[0] != v15 )
        return j_j___libc_free_0(v14[0], v15[0] + 1LL);
    }
  }
  return result;
}
