// Function: sub_324C7E0
// Address: 0x324c7e0
//
void __fastcall sub_324C7E0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // r14
  __int64 v7; // r15
  unsigned __int64 v8; // rax

  if ( a3 )
  {
    v3 = *(_BYTE *)(a3 - 16);
    if ( (v3 & 2) != 0 )
    {
      v4 = *(__int64 **)(a3 - 32);
      v5 = *(unsigned int *)(a3 - 24);
    }
    else
    {
      v4 = (__int64 *)(a3 - 16 - 8LL * ((v3 >> 2) & 0xF));
      v5 = (*(_WORD *)(a3 - 16) >> 6) & 0xF;
    }
    v6 = &v4[v5];
    while ( v6 != v4 )
    {
      v7 = *v4++;
      v8 = sub_324C6D0(a1, 49, a2, 0);
      sub_32495E0(a1, v8, v7, 73);
    }
  }
}
