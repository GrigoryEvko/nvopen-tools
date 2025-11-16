// Function: sub_B15890
// Address: 0xb15890
//
void __fastcall sub_B15890(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // al
  __int64 *v4; // rdx

  *a1 = 0;
  a1[1] = 0;
  if ( a2 )
  {
    v2 = a2;
    if ( *(_BYTE *)a2 != 16 )
    {
      v3 = *(_BYTE *)(a2 - 16);
      if ( (v3 & 2) != 0 )
        v4 = *(__int64 **)(a2 - 32);
      else
        v4 = (__int64 *)(a2 - 16 - 8LL * ((v3 >> 2) & 0xF));
      v2 = *v4;
    }
    *a1 = v2;
    *((_DWORD *)a1 + 2) = *(_DWORD *)(a2 + 20);
  }
}
