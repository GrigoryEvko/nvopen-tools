// Function: sub_13DB920
// Address: 0x13db920
//
__int64 __fastcall sub_13DB920(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, unsigned int a5)
{
  __int64 **v5; // rax

  v5 = sub_13D9330(a1, a2, a3, a4, a5);
  if ( v5 && *((_BYTE *)v5 + 16) <= 0x10u )
    return sub_1596070(v5);
  else
    return 0;
}
