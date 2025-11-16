// Function: sub_27B98E0
// Address: 0x27b98e0
//
__int64 __fastcall sub_27B98E0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rsi
  unsigned int v3; // r8d
  __int64 v4; // rdi
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  unsigned int v8; // r8d

  v2 = *a2;
  v3 = 1;
  if ( v2 == **(_QWORD **)a1 )
    return v3;
  v4 = *(_QWORD *)(a1 + 8);
  v3 = *(unsigned __int8 *)(v4 + 84);
  if ( !(_BYTE)v3 )
  {
    LOBYTE(v8) = sub_C8CA60(v4 + 56, v2) != 0;
    return v8;
  }
  v5 = *(_QWORD **)(v4 + 64);
  v6 = &v5[*(unsigned int *)(v4 + 76)];
  if ( v5 != v6 )
  {
    while ( v2 != *v5 )
    {
      if ( v6 == ++v5 )
        return 0;
    }
    return v3;
  }
  return 0;
}
