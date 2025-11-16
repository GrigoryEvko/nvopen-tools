// Function: sub_210E320
// Address: 0x210e320
//
__int64 __fastcall sub_210E320(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax
  __int64 *v3; // r13
  __int64 v4; // r12
  __int64 v5; // rdi

  v1 = *(__int64 **)(a1 + 400);
  result = *(unsigned int *)(a1 + 408);
  v3 = &v1[result];
  if ( v3 != v1 )
  {
    while ( 1 )
    {
      v5 = *v1;
      if ( (*(_BYTE *)(*v1 + 23) & 0x40) != 0 )
      {
        v4 = *(_QWORD *)(*(_QWORD *)(v5 - 8) + 24LL);
        if ( *(_BYTE *)(v4 + 16) > 0x17u )
          goto LABEL_4;
LABEL_8:
        result = sub_15F20C0((_QWORD *)v5);
LABEL_9:
        if ( v3 == ++v1 )
          return result;
      }
      else
      {
        v4 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) + 24);
        if ( *(_BYTE *)(v4 + 16) <= 0x17u )
          goto LABEL_8;
LABEL_4:
        result = sub_15F20C0((_QWORD *)v5);
        if ( !*(_QWORD *)(v4 + 8) )
        {
          result = sub_15F20C0((_QWORD *)v4);
          goto LABEL_9;
        }
        if ( v3 == ++v1 )
          return result;
      }
    }
  }
  return result;
}
