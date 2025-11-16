// Function: sub_102D1C0
// Address: 0x102d1c0
//
__int64 __fastcall sub_102D1C0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  unsigned int v5; // r8d
  __int64 v6; // rax

  if ( !*(_BYTE *)(a1 + 96) )
  {
    v6 = sub_B43CB0(a2);
    sub_102D140(a1, v6);
    *(_BYTE *)(a1 + 96) = 1;
  }
  if ( *(_BYTE *)(a1 + 28) )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v3 = &v2[*(unsigned int *)(a1 + 20)];
    if ( v2 == v3 )
    {
      return 0;
    }
    else
    {
      while ( a2 != *v2 )
      {
        if ( v3 == ++v2 )
          return 0;
      }
      return *(unsigned __int8 *)(a1 + 28);
    }
  }
  else
  {
    LOBYTE(v5) = sub_C8CA60(a1, a2) != 0;
    return v5;
  }
}
