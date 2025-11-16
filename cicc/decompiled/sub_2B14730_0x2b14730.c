// Function: sub_2B14730
// Address: 0x2b14730
//
__int64 __fastcall sub_2B14730(__int64 a1)
{
  __int64 v3; // rax
  __int64 v4; // r10
  unsigned __int8 **v5; // rdi
  unsigned __int8 **v6; // rax
  unsigned __int8 **v7; // r10

  if ( *(_BYTE *)a1 > 0x1Cu
    && ((unsigned __int8)sub_991AB0((char *)a1)
     || ((v3 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF), (*(_BYTE *)(a1 + 7) & 0x40) != 0)
       ? (unsigned __int8 **)(v5 = *(unsigned __int8 ***)(a1 - 8), v4 = (__int64)&v5[v3])
       : (v4 = a1, v5 = (unsigned __int8 **)(a1 - v3 * 8)),
         v6 = sub_2B12F20(v5, v4, a1),
         v7 != v6)) )
  {
    return 0;
  }
  else
  {
    return sub_2B099C0(a1);
  }
}
