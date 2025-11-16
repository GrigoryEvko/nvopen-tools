// Function: sub_2EA49F0
// Address: 0x2ea49f0
//
__int64 __fastcall sub_2EA49F0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  __int64 *v3; // r12
  __int64 *v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rax
  _QWORD *v7; // rdx

  v1 = 0;
  v2 = **(_QWORD **)(a1 + 32);
  v3 = *(__int64 **)(v2 + 64);
  v4 = &v3[*(unsigned int *)(v2 + 72)];
  if ( v3 != v4 )
  {
    while ( 1 )
    {
      v5 = *v3;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v6 = *(_QWORD **)(a1 + 64);
        v7 = &v6[*(unsigned int *)(a1 + 76)];
        if ( v6 == v7 )
          goto LABEL_9;
        while ( v5 != *v6 )
        {
          if ( v7 == ++v6 )
            goto LABEL_9;
        }
        if ( v1 )
          return 0;
      }
      else
      {
        if ( !sub_C8CA60(a1 + 56, *v3) )
          goto LABEL_9;
        if ( v1 )
          return 0;
      }
      v1 = v5;
LABEL_9:
      if ( v4 == ++v3 )
        return v1;
    }
  }
  return 0;
}
