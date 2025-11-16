// Function: sub_26F5F30
// Address: 0x26f5f30
//
__int64 __fastcall sub_26F5F30(__int64 a1)
{
  unsigned __int64 v1; // rdi
  __int64 *v2; // rdx
  __int64 *v3; // rsi
  __int64 v4; // rax
  int v5; // ecx

  if ( !(_BYTE)qword_4FF90E8 )
  {
    v1 = a1 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v1 )
    {
      v2 = *(__int64 **)(v1 + 24);
      v3 = *(__int64 **)(v1 + 32);
      if ( v2 != v3 )
      {
        while ( 1 )
        {
          v4 = *v2;
          if ( *(char *)(*v2 + 12) >= 0 )
            break;
          v5 = *(_DWORD *)(v4 + 8);
          if ( !v5 )
          {
            v4 = *(_QWORD *)(v4 + 64);
            v5 = *(_DWORD *)(v4 + 8);
          }
          if ( v5 != 1 || (*(_BYTE *)(v4 + 61) & 2) == 0 )
            break;
          if ( v3 == ++v2 )
            return 1;
        }
      }
    }
  }
  return 0;
}
