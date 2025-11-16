// Function: sub_15640D0
// Address: 0x15640d0
//
__int64 __fastcall sub_15640D0(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rax
  _BYTE *v3; // rdi
  _QWORD *v4; // rax
  unsigned __int64 v5; // rdx

  v1 = 0;
  if ( a1 )
  {
    if ( *(_BYTE *)a1 == 4 )
    {
      v2 = *(unsigned int *)(a1 + 8);
      if ( (_DWORD)v2 )
      {
        v3 = *(_BYTE **)(a1 - 8 * v2);
        if ( v3 )
        {
          if ( !*v3 )
          {
            v4 = (_QWORD *)sub_161E970(v3);
            if ( v5 > 0xF )
              LOBYTE(v1) = (*v4 ^ 0x6365762E6D766C6CLL | v4[1] ^ 0x2E72657A69726F74LL) == 0;
          }
        }
      }
    }
  }
  return v1;
}
