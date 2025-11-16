// Function: sub_1054B40
// Address: 0x1054b40
//
__int64 __fastcall sub_1054B40(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rsi
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  unsigned int v7; // r8d

  v2 = *a1;
  v3 = *(_QWORD *)(a2 + 48);
  if ( *(_BYTE *)(v2 + 28) )
  {
    v4 = *(_QWORD **)(v2 + 8);
    v5 = &v4[*(unsigned int *)(v2 + 20)];
    if ( v4 == v5 )
    {
      return 0;
    }
    else
    {
      while ( v3 != *v4 )
      {
        if ( v5 == ++v4 )
          return 0;
      }
      return *(unsigned __int8 *)(v2 + 28);
    }
  }
  else
  {
    LOBYTE(v7) = sub_C8CA60(v2, v3) != 0;
    return v7;
  }
}
