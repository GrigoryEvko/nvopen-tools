// Function: sub_645520
// Address: 0x645520
//
__int64 __fastcall sub_645520(__int64 *a1)
{
  int v2; // eax
  __int64 v3; // rdi
  __int64 v4; // rax
  char v5; // r12
  __int64 v6; // rax
  unsigned int v7; // r12d
  __int64 result; // rax

  v2 = sub_8D3410(*a1);
  v3 = *a1;
  if ( v2 )
  {
    v4 = *a1;
    if ( *(_BYTE *)(v3 + 140) == 12 )
    {
      do
        v4 = *(_QWORD *)(v4 + 160);
      while ( *(_BYTE *)(v4 + 140) == 12 );
    }
    v5 = *(_BYTE *)(v4 + 168);
    v6 = sub_8D4050(v3);
    v7 = v5 & 0x7F;
    result = sub_72D2E0(v6, 0);
    *a1 = result;
    if ( v7 )
    {
      result = sub_73C570(result, v7, -1);
      *a1 = result;
    }
  }
  else
  {
    result = sub_8D2310(v3);
    if ( (_DWORD)result )
    {
      result = sub_72D2E0(*a1, 0);
      *a1 = result;
    }
  }
  return result;
}
