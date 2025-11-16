// Function: sub_2ECD8F0
// Address: 0x2ecd8f0
//
void __fastcall sub_2ECD8F0(__int64 a1, __int64 a2, unsigned __int8 (__fastcall *a3)(__int64 *, _QWORD *))
{
  _QWORD *v3; // rbx
  __int64 *v4; // r12
  __int64 v6; // [rsp-40h] [rbp-40h]

  if ( a2 != a1 )
  {
    v3 = *(_QWORD **)a1;
    v4 = *(__int64 **)a2;
    if ( *(_QWORD *)a1 == a1 )
    {
LABEL_9:
      if ( (__int64 *)a2 != v4 )
        sub_2208C50(a1, (__int64)v4, a2);
    }
    else if ( (__int64 *)a2 != v4 )
    {
      do
      {
        if ( a3(v4 + 2, v3 + 2) )
        {
          v6 = *v4;
          sub_2208C50((__int64)v3, (__int64)v4, *v4);
          v4 = (__int64 *)v6;
          if ( v3 == (_QWORD *)a1 )
            goto LABEL_9;
        }
        else
        {
          v3 = (_QWORD *)*v3;
          if ( v3 == (_QWORD *)a1 )
            goto LABEL_9;
        }
      }
      while ( v4 != (__int64 *)a2 );
    }
    *(_QWORD *)(a1 + 16) += *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a2 + 16) = 0;
  }
}
