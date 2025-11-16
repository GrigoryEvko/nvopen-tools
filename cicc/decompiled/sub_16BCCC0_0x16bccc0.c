// Function: sub_16BCCC0
// Address: 0x16bccc0
//
__int64 *__fastcall sub_16BCCC0(__int64 *a1, unsigned int a2, __int64 a3, _BYTE *a4)
{
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // rbx
  _BYTE *v9; // [rsp+0h] [rbp-40h] BYREF
  __int16 v10; // [rsp+10h] [rbp-30h]

  v5 = *a4 == 0;
  v10 = 257;
  if ( !v5 )
  {
    v9 = a4;
    LOBYTE(v10) = 3;
  }
  v6 = sub_22077B0(56);
  v7 = v6;
  if ( v6 )
    sub_16BCC70(v6, (__int64)&v9, a2, a3);
  *a1 = v7 | 1;
  return a1;
}
