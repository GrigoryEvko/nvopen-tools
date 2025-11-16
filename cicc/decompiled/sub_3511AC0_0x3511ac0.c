// Function: sub_3511AC0
// Address: 0x3511ac0
//
char *__fastcall sub_3511AC0(
        __int64 *src,
        __int64 *a2,
        __int64 *a3,
        __int64 *a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 *v8; // r15
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  char *v13; // r8
  __int64 v16; // [rsp+10h] [rbp-40h]
  unsigned int v17; // [rsp+1Ch] [rbp-34h]

  v8 = a3;
  v10 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v16 = *v8;
      v17 = sub_2E441D0(*(_QWORD *)(a7 + 528), *a8, *v10);
      if ( v17 < (unsigned int)sub_2E441D0(*(_QWORD *)(a7 + 528), *a8, v16) )
      {
        v11 = *v8;
        ++a5;
        ++v8;
        *(a5 - 1) = v11;
        if ( v10 == a2 )
          break;
      }
      else
      {
        v12 = *v10;
        ++a5;
        ++v10;
        *(a5 - 1) = v12;
        if ( v10 == a2 )
          break;
      }
    }
    while ( v8 != a4 );
  }
  if ( a2 != v10 )
    memmove(a5, v10, (char *)a2 - (char *)v10);
  v13 = (char *)a5 + (char *)a2 - (char *)v10;
  if ( a4 != v8 )
    v13 = (char *)memmove(v13, v8, (char *)a4 - (char *)v8);
  return &v13[(char *)a4 - (char *)v8];
}
