// Function: sub_13E0EE0
// Address: 0x13e0ee0
//
unsigned __int8 *__fastcall sub_13E0EE0(__int64 a1, unsigned __int8 *a2, __int64 a3, char a4, _QWORD *a5, int a6)
{
  unsigned __int8 *result; // rax
  __int64 v10; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-38h]
  _BYTE *v12; // [rsp+20h] [rbp-30h]
  unsigned int v13; // [rsp+28h] [rbp-28h]

  result = sub_13E0AE0(a1, a2, a3, a5, a6);
  if ( result )
    return result;
  if ( a2 == (unsigned __int8 *)a3 )
    return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)a2);
  if ( a2[16] == 9 )
  {
    result = a2;
    if ( a4 )
      return result;
    return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)a2);
  }
  if ( !a4 )
    return result;
  sub_14C2530((unsigned int)&v10, (_DWORD)a2, *a5, 0, a5[3], a5[4], a5[2], 0);
  result = 0;
  if ( v13 > 0x40 )
  {
    if ( (*v12 & 1) != 0 )
    {
      j_j___libc_free_0_0(v12);
      goto LABEL_8;
    }
    j_j___libc_free_0_0(v12);
    result = 0;
  }
  else if ( ((unsigned __int8)v12 & 1) != 0 )
  {
LABEL_8:
    if ( v11 > 0x40 )
    {
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    return a2;
  }
  if ( v11 > 0x40 && v10 )
  {
    j_j___libc_free_0_0(v10);
    return 0;
  }
  return result;
}
