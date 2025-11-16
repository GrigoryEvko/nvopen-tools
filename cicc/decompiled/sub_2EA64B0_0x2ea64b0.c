// Function: sub_2EA64B0
// Address: 0x2ea64b0
//
__int64 __fastcall sub_2EA64B0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD *v4; // rdx

  v1 = **(_QWORD **)(a1 + 32);
  if ( v1 != (*(_QWORD *)(*(_QWORD *)(v1 + 32) + 320LL) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v2 = *(_QWORD *)(v1 + 8);
    if ( !*(_BYTE *)(a1 + 84) )
      goto LABEL_9;
LABEL_3:
    v3 = *(_QWORD **)(a1 + 64);
    v4 = &v3[*(unsigned int *)(a1 + 76)];
    if ( v3 != v4 )
    {
      while ( v2 != *v3 )
      {
        if ( v4 == ++v3 )
          return v1;
      }
      do
      {
        v1 = *(_QWORD *)(v2 + 8);
        if ( v2 == v1 )
          break;
        v1 = v2;
        v2 = *(_QWORD *)(v2 + 8);
        if ( *(_BYTE *)(a1 + 84) )
          goto LABEL_3;
LABEL_9:
        ;
      }
      while ( sub_C8CA60(a1 + 56, v2) );
    }
  }
  return v1;
}
