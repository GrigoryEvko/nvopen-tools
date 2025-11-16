// Function: sub_1689880
// Address: 0x1689880
//
__int64 *__fastcall sub_1689880(__int64 *a1, _BYTE *a2, unsigned int a3, __int64 a4)
{
  bool v5; // zf
  __int64 v7; // rax
  __int64 v8; // rbx
  _BYTE *v10; // [rsp+0h] [rbp-40h] BYREF
  __int16 v11; // [rsp+10h] [rbp-30h]

  v5 = *a2 == 0;
  v11 = 257;
  if ( !v5 )
  {
    v10 = a2;
    LOBYTE(v11) = 3;
  }
  v7 = sub_22077B0(56);
  v8 = v7;
  if ( v7 )
    sub_16BCC70(v7, &v10, a3, a4);
  *a1 = v8 | 1;
  return a1;
}
