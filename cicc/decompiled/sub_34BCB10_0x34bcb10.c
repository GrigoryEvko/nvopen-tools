// Function: sub_34BCB10
// Address: 0x34bcb10
//
void __fastcall sub_34BCB10(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int8 (__fastcall *a3)(__int64, __int64 *, unsigned __int64 *),
        __int64 a4)
{
  __int64 *v5; // r14
  unsigned __int64 *v6; // r15
  unsigned __int64 *i; // rbx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  __int64 *v13; // [rsp+8h] [rbp-48h]

  if ( a1 != a2 && a2 != (unsigned __int64 *)(*a2 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v5 = (__int64 *)a2[1];
    v6 = (unsigned __int64 *)a1[1];
    v13 = v5;
    if ( a1 == v6 )
    {
LABEL_16:
      if ( a2 != a1 && a2 != (unsigned __int64 *)v13 )
      {
        v11 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 8) = a2;
        *a2 = *a2 & 7 | *v5 & 0xFFFFFFFFFFFFFFF8LL;
        v12 = *a1;
        *(_QWORD *)(v11 + 8) = a1;
        v12 &= 0xFFFFFFFFFFFFFFF8LL;
        *v5 = v12 | *v5 & 7;
        *(_QWORD *)(v12 + 8) = v13;
        *a1 = v11 | *a1 & 7;
      }
    }
    else
    {
      while ( 1 )
      {
        while ( !a3(a4, v5, v6) )
        {
          v6 = (unsigned __int64 *)v6[1];
          if ( a1 == v6 )
            goto LABEL_16;
        }
        for ( i = (unsigned __int64 *)v5[1]; a2 != i; i = (unsigned __int64 *)i[1] )
        {
          if ( !a3(a4, (__int64 *)i, v6) )
            break;
        }
        if ( i != v6 && i != (unsigned __int64 *)v13 )
        {
          v9 = *i & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 8) = i;
          *i = *i & 7 | *v5 & 0xFFFFFFFFFFFFFFF8LL;
          v10 = *v6;
          *(_QWORD *)(v9 + 8) = v6;
          v10 &= 0xFFFFFFFFFFFFFFF8LL;
          *v5 = v10 | *v5 & 7;
          *(_QWORD *)(v10 + 8) = v13;
          *v6 = v9 | *v6 & 7;
        }
        if ( a2 == i )
          break;
        v13 = (__int64 *)i;
        v6 = (unsigned __int64 *)v6[1];
        v5 = (__int64 *)i;
        if ( a1 == v6 )
          goto LABEL_16;
      }
    }
  }
}
