// Function: sub_7E0F10
// Address: 0x7e0f10
//
__int64 __fastcall sub_7E0F10(__int64 a1)
{
  __int64 result; // rax
  _QWORD *i; // r13
  _QWORD *j; // r14
  _QWORD *k; // r15
  _QWORD *m; // rcx
  _QWORD *n; // r12
  _QWORD *ii; // rbx
  _QWORD *jj; // r8
  _QWORD *kk; // rdx
  __int64 mm; // rax
  _QWORD *v11; // [rsp+0h] [rbp-50h]
  _QWORD *v12; // [rsp+8h] [rbp-48h]
  _QWORD *v13; // [rsp+10h] [rbp-40h]
  __int64 *v14; // [rsp+18h] [rbp-38h]

  for ( result = *(_QWORD *)(a1 + 104); result; result = *(_QWORD *)(result + 112) )
  {
    if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
      *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
  }
  for ( i = *(_QWORD **)(a1 + 160); i; i = (_QWORD *)*i )
  {
    for ( result = i[13]; result; result = *(_QWORD *)(result + 112) )
    {
      if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
        *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
    }
    for ( j = (_QWORD *)i[20]; j; j = (_QWORD *)*j )
    {
      for ( result = j[13]; result; result = *(_QWORD *)(result + 112) )
      {
        if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
          *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
      }
      for ( k = (_QWORD *)j[20]; k; k = (_QWORD *)*k )
      {
        for ( result = k[13]; result; result = *(_QWORD *)(result + 112) )
        {
          if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
            *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
        }
        for ( m = (_QWORD *)k[20]; m; m = (_QWORD *)*m )
        {
          for ( result = m[13]; result; result = *(_QWORD *)(result + 112) )
          {
            if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
              *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
          }
          for ( n = (_QWORD *)m[20]; n; n = (_QWORD *)*n )
          {
            for ( result = n[13]; result; result = *(_QWORD *)(result + 112) )
            {
              if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
                *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
            }
            for ( ii = (_QWORD *)n[20]; ii; ii = (_QWORD *)*ii )
            {
              for ( result = ii[13]; result; result = *(_QWORD *)(result + 112) )
              {
                if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
                  *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
              }
              for ( jj = (_QWORD *)ii[20]; jj; jj = (_QWORD *)*jj )
              {
                for ( result = jj[13]; result; result = *(_QWORD *)(result + 112) )
                {
                  if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
                    *(_QWORD *)(*(_QWORD *)(result + 168) + 184LL) = 0;
                }
                for ( kk = (_QWORD *)jj[20]; kk; kk = (_QWORD *)*kk )
                {
                  for ( mm = kk[13]; mm; mm = *(_QWORD *)(mm + 112) )
                  {
                    if ( (unsigned __int8)(*(_BYTE *)(mm + 140) - 9) <= 2u )
                      *(_QWORD *)(*(_QWORD *)(mm + 168) + 184LL) = 0;
                  }
                  result = kk[20];
                  if ( result )
                  {
                    do
                    {
                      v11 = kk;
                      v12 = jj;
                      v13 = m;
                      v14 = (__int64 *)result;
                      sub_7E0F10(result);
                      m = v13;
                      jj = v12;
                      kk = v11;
                      result = *v14;
                    }
                    while ( *v14 );
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
