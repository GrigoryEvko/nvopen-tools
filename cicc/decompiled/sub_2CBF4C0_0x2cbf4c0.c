// Function: sub_2CBF4C0
// Address: 0x2cbf4c0
//
__int64 __fastcall sub_2CBF4C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned int v4; // r8d
  __int64 v5; // rcx

  v3 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == a1 + 48 )
    goto LABEL_13;
  if ( !v3 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
LABEL_13:
    BUG();
  v4 = 0;
  if ( *(_BYTE *)(v3 - 24) == 31 && (*(_DWORD *)(v3 - 20) & 0x7FFFFFF) == 3 )
  {
    v5 = *(_QWORD *)(v3 - 56);
    if ( a2 == v5 && a3 == *(_QWORD *)(v3 - 88) )
    {
      return 1;
    }
    else
    {
      v4 = 0;
      if ( a3 == v5 )
        LOBYTE(v4) = *(_QWORD *)(v3 - 88) == a2;
    }
  }
  return v4;
}
