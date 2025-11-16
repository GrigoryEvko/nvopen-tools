// Function: sub_13FD440
// Address: 0x13fd440
//
__int64 __fastcall sub_13FD440(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rcx
  _QWORD *v8; // rax
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v1 = sub_13FD000(a1);
  if ( !v1 )
    return 0;
  v2 = v1;
  v10 = *(_QWORD *)(a1 + 40);
  v11 = *(_QWORD *)(a1 + 32);
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      v3 = *(_QWORD *)(*(_QWORD *)v11 + 48LL);
      v4 = *(_QWORD *)v11 + 40LL;
      if ( v3 != v4 )
        break;
LABEL_14:
      v11 += 8;
      if ( v10 == v11 )
        return 1;
    }
    while ( 1 )
    {
      v5 = v3 - 24;
      if ( !v3 )
        v5 = 0;
      if ( (unsigned __int8)sub_15F2ED0(v5) )
      {
        if ( !*(_QWORD *)(v5 + 48) )
          goto LABEL_18;
      }
      else
      {
        if ( !(unsigned __int8)sub_15F3040(v5) )
          goto LABEL_13;
        if ( !*(_QWORD *)(v5 + 48) )
        {
LABEL_18:
          if ( *(__int16 *)(v5 + 18) >= 0 )
            return 0;
        }
      }
      v6 = sub_1625790(v5, 10);
      v7 = (_QWORD *)v6;
      if ( !v6 )
        return 0;
      v8 = (_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8));
      if ( v8 == v7 )
        return 0;
      while ( v2 != *v8 )
      {
        if ( v7 == ++v8 )
          return 0;
      }
LABEL_13:
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 == v3 )
        goto LABEL_14;
    }
  }
  return 1;
}
