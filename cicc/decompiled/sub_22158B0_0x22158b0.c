// Function: sub_22158B0
// Address: 0x22158b0
//
volatile signed __int32 **__fastcall sub_22158B0(volatile signed __int32 **a1, volatile signed __int32 *a2, size_t a3)
{
  volatile signed __int32 *v4; // rax
  __int64 v5; // rdx
  volatile signed __int32 *v8; // rdi
  size_t v9; // [rsp+8h] [rbp-10h]
  size_t v10; // [rsp+8h] [rbp-10h]

  v4 = *a1;
  v5 = *((_QWORD *)*a1 - 3);
  if ( a3 > 0x3FFFFFFFFFFFFFF9LL )
    sub_4262D8((__int64)"basic_string::assign");
  if ( v4 > a2 || (volatile signed __int32 *)((char *)v4 + v5) < a2 )
    return sub_2215840(a1, 0, v5, a2, a3);
  if ( *((int *)v4 - 2) > 0 )
  {
    v5 = *((_QWORD *)*a1 - 3);
    return sub_2215840(a1, 0, v5, a2, a3);
  }
  v8 = *a1;
  if ( a3 <= (char *)a2 - (char *)v8 )
  {
    if ( a3 != 1 )
    {
      if ( a3 )
      {
        v10 = a3;
        memcpy((void *)v8, (const void *)a2, a3);
        v8 = *a1;
        a3 = v10;
      }
      goto LABEL_12;
    }
LABEL_18:
    *(_BYTE *)v8 = *(_BYTE *)a2;
    v8 = *a1;
    goto LABEL_12;
  }
  if ( a2 != v8 )
  {
    if ( a3 != 1 )
    {
      if ( a3 )
      {
        v9 = a3;
        memmove((void *)v8, (const void *)a2, a3);
        v8 = *a1;
        a3 = v9;
      }
      goto LABEL_12;
    }
    goto LABEL_18;
  }
LABEL_12:
  if ( v8 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
  {
    *((_DWORD *)v8 - 2) = 0;
    *((_QWORD *)v8 - 3) = a3;
    *((_BYTE *)v8 + a3) = 0;
  }
  return a1;
}
