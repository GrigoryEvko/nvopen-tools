// Function: sub_2B21B90
// Address: 0x2b21b90
//
unsigned __int8 *__fastcall sub_2B21B90(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 ****a6)
{
  __int64 v6; // r10
  __int64 v9; // rax
  char v10; // r9
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // r12
  unsigned __int8 **v14; // rdi
  __int64 v15; // rsi
  _BYTE **v16; // rax
  __int64 v17; // [rsp+8h] [rbp-28h]

  v6 = a1;
  v9 = *((unsigned int *)a6 + 2);
  v10 = 1;
  if ( v9 != 2 )
  {
    v10 = 0;
    if ( v9 == 1 )
    {
      v17 = a3;
      v14 = **a6;
      v15 = (__int64)&v14[*((unsigned int *)*a6 + 2)];
      v16 = sub_2B0AAE0(v14, v15);
      a3 = v17;
      v10 = v15 != (_QWORD)v16;
    }
  }
  v11 = (unsigned __int8 *)sub_2B21610(v6, a2, a3, a4, a5, v10);
  v12 = v11;
  if ( a2 - 6 <= 3 && *v11 == 86 )
  {
    sub_F70480(*((unsigned __int8 **)v11 - 12), **a6, *((unsigned int *)*a6 + 2), 0, 0);
    sub_F70480(v12, (*a6)[18], *((unsigned int *)*a6 + 38), 0, 0);
    return v12;
  }
  else
  {
    sub_F70480(v11, **a6, *((unsigned int *)*a6 + 2), 0, 0);
    return v12;
  }
}
