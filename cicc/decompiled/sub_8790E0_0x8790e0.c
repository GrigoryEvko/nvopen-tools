// Function: sub_8790E0
// Address: 0x8790e0
//
void __fastcall sub_8790E0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 *v3; // rdx
  __int64 *v4; // rsi
  __int64 *v5; // rax
  __int64 *v6; // rax

  if ( (*((_BYTE *)a1 + 81) & 0x20) == 0 && (unsigned __int8)(*((_BYTE *)a1 + 80) - 14) > 1u )
  {
    v1 = *a1;
    v2 = a1[1];
    v3 = *(__int64 **)(*a1 + 24);
    if ( v3 == a1 )
    {
      *(_QWORD *)(v1 + 24) = v2;
      goto LABEL_12;
    }
    v4 = *(__int64 **)(v1 + 32);
    if ( v4 == a1 )
    {
      *(_QWORD *)(v1 + 32) = v2;
      goto LABEL_12;
    }
    if ( v3 )
    {
      v5 = (__int64 *)v3[1];
      if ( a1 == v5 )
      {
LABEL_10:
        if ( v5 )
        {
LABEL_11:
          v3[1] = v2;
          goto LABEL_12;
        }
      }
      else if ( v5 )
      {
        while ( 1 )
        {
          v3 = v5;
          v5 = (__int64 *)v5[1];
          if ( !v5 )
            break;
          if ( a1 == v5 )
            goto LABEL_10;
        }
      }
    }
    if ( v4 )
    {
      v6 = (__int64 *)v4[1];
      if ( a1 != v6 && v6 )
      {
        do
        {
          v3 = v6;
          v6 = (__int64 *)v6[1];
        }
        while ( v6 && a1 != v6 );
      }
      else
      {
        v3 = v4;
      }
    }
    goto LABEL_11;
  }
LABEL_12:
  a1[1] = 0;
}
