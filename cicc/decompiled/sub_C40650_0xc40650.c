// Function: sub_C40650
// Address: 0xc40650
//
__int64 __fastcall sub_C40650(__int64 a1, __int64 *a2, unsigned int a3, unsigned int a4, unsigned __int8 a5)
{
  _DWORD *v7; // rax
  _DWORD *v8; // r12
  __int64 result; // rax
  _QWORD *i; // rbx
  __int64 v12; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-58h]
  _DWORD *v14; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+28h] [rbp-48h]

  sub_C3E660((__int64)&v12, a1);
  v7 = sub_C33340();
  v8 = v7;
  if ( v7 == dword_3F65580 )
    sub_C3C640(&v14, (__int64)v7, &v12);
  else
    sub_C3B160((__int64)&v14, dword_3F65580, &v12);
  if ( v14 == v8 )
    sub_C40650(&v14, a2, a3, a4, a5);
  else
    sub_C35AD0((__int64)&v14, a2, a3, a4, a5);
  if ( v14 == v8 )
  {
    result = (__int64)v15;
    if ( v15 )
    {
      for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v8 == (_DWORD *)*i )
            break;
          sub_C338F0((__int64)i);
          if ( v15 == i )
            goto LABEL_13;
        }
      }
LABEL_13:
      result = j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    result = sub_C338F0((__int64)&v14);
  }
  if ( v13 > 0x40 )
  {
    if ( v12 )
      return j_j___libc_free_0_0(v12);
  }
  return result;
}
