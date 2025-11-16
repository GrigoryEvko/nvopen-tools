// Function: sub_3511E50
// Address: 0x3511e50
//
void __fastcall sub_3511E50(__int64 *a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  unsigned int v8; // ebx
  __int64 *v9; // r14
  __int64 v10; // r15
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __int64 *v13; // [rsp+10h] [rbp-40h]
  __int64 *v14; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v13 = a1 + 1;
    do
    {
      while ( 1 )
      {
        v7 = *v13;
        v8 = sub_2E441D0(*(_QWORD *)(a3 + 528), *a4, *a1);
        if ( v8 < (unsigned int)sub_2E441D0(*(_QWORD *)(a3 + 528), *a4, v7) )
          break;
        v9 = v13;
        v10 = *v13;
        while ( 1 )
        {
          v11 = *(v9 - 1);
          v14 = v9--;
          v12 = sub_2E441D0(*(_QWORD *)(a3 + 528), *a4, v11);
          if ( v12 >= (unsigned int)sub_2E441D0(*(_QWORD *)(a3 + 528), *a4, v10) )
            break;
          v9[1] = *v9;
        }
        ++v13;
        *v14 = v10;
        if ( a2 == v13 )
          return;
      }
      v6 = *v13;
      if ( a1 != v13 )
        memmove(a1 + 1, a1, (char *)v13 - (char *)a1);
      ++v13;
      *a1 = v6;
    }
    while ( a2 != v13 );
  }
}
