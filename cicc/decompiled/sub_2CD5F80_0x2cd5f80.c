// Function: sub_2CD5F80
// Address: 0x2cd5f80
//
__int64 __fastcall sub_2CD5F80(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rsi
  unsigned int v4; // r8d
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  unsigned int v8; // r8d

  v2 = *a1;
  v3 = *(_QWORD *)(a2 + 24);
  v4 = *(unsigned __int8 *)(v2 + 28);
  if ( (_BYTE)v4 )
  {
    v5 = *(_QWORD **)(v2 + 8);
    v6 = &v5[*(unsigned int *)(v2 + 20)];
    if ( v5 != v6 )
    {
      while ( v3 != *v5 )
      {
        if ( v6 == ++v5 )
          return v4;
      }
      return 0;
    }
    return v4;
  }
  else
  {
    LOBYTE(v8) = sub_C8CA60(v2, v3) == 0;
    return v8;
  }
}
