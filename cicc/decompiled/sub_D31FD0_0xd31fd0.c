// Function: sub_D31FD0
// Address: 0xd31fd0
//
__int64 __fastcall sub_D31FD0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-58h]
  __int64 v11; // [rsp+10h] [rbp-50h]

  v10 = *(_QWORD *)(a1 + 40);
  if ( v10 != *(_QWORD *)(a1 + 32) )
  {
    v11 = *(_QWORD *)(a1 + 32);
    do
    {
      v7 = *(_QWORD *)v11 + 48LL;
      if ( v7 != *(_QWORD *)(*(_QWORD *)v11 + 56LL) )
      {
        v8 = *(_QWORD *)(*(_QWORD *)v11 + 56LL);
        do
        {
          if ( !v8 )
            BUG();
          if ( *(_BYTE *)(v8 - 24) == 61 )
          {
            if ( !(unsigned __int8)sub_D30800(v8 - 24, a1, a2, a3, a4, a5) )
              return 0;
          }
          else if ( (unsigned __int8)sub_B46420(v8 - 24)
                 || (unsigned __int8)sub_B46490(v8 - 24)
                 || (unsigned __int8)sub_B46790((unsigned __int8 *)(v8 - 24), 0) )
          {
            return 0;
          }
          v8 = *(_QWORD *)(v8 + 8);
        }
        while ( v7 != v8 );
      }
      v11 += 8;
    }
    while ( v10 != v11 );
  }
  return 1;
}
