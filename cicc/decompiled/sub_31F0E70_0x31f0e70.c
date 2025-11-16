// Function: sub_31F0E70
// Address: 0x31f0e70
//
void (*__fastcall sub_31F0E70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8))()
{
  __int64 v8; // r12
  __int64 (__fastcall *v9)(__int64, __int64, __int64); // rbx
  __int64 v10; // rdx

  if ( *(_BYTE *)(a1 + 976) )
    return sub_31F0D70(a1, a7, 0);
  v8 = *(_QWORD *)(a1 + 224);
  v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v8 + 536LL);
  v10 = (unsigned int)sub_31DF6B0(a1);
  return (void (*)())v9(v8, a8, v10);
}
