// Function: sub_16E35D0
// Address: 0x16e35d0
//
__int64 __fastcall sub_16E35D0(__int64 a1, const char *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // r13
  size_t v4; // r14

  v2 = *(unsigned __int8 *)(a1 + 272);
  if ( (_BYTE)v2 )
  {
    return 0;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 264);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v3 + 8) + 32LL) - 1) <= 1 )
    {
      v4 = *(_QWORD *)(v3 + 24);
      if ( !a2 )
      {
        if ( v4 )
          return v2;
        goto LABEL_9;
      }
      if ( v4 == strlen(a2) && (!v4 || !memcmp(*(const void **)(v3 + 16), a2, v4)) )
      {
LABEL_9:
        *(_BYTE *)(a1 + 272) = 1;
        return 1;
      }
    }
  }
  return v2;
}
