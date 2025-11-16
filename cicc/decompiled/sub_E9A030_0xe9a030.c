// Function: sub_E9A030
// Address: 0xe9a030
//
__int64 __fastcall sub_E9A030(__int64 *a1)
{
  __int64 v1; // rbp
  __int64 result; // rax
  __int64 v3; // rax
  void (*v4)(); // rcx
  void (*(__fastcall *v5)(__int64 *, unsigned __int64, unsigned int))(); // rcx
  char *v6; // rsi
  unsigned __int64 v7; // rdx
  _QWORD v8[4]; // [rsp-38h] [rbp-38h] BYREF
  char v9; // [rsp-18h] [rbp-18h]
  char v10; // [rsp-17h] [rbp-17h]
  __int64 v11; // [rsp-8h] [rbp-8h]

  result = a1[1];
  if ( *(_BYTE *)(result + 1906) == 1 )
  {
    v11 = v1;
    v3 = *a1;
    v4 = *(void (**)())(*a1 + 120);
    v10 = 1;
    v8[0] = "DWARF64 Mark";
    v9 = 3;
    if ( v4 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v4)(a1, v8, 1);
      v3 = *a1;
    }
    v5 = *(void (*(__fastcall **)(__int64 *, unsigned __int64, unsigned int))())(v3 + 536);
    if ( v5 == sub_E97D30 )
    {
      v6 = (char *)v8 + 4;
      v7 = 0xFFFFFFFF00000000LL;
      if ( *(_BYTE *)(*(_QWORD *)(a1[1] + 152) + 16LL) )
      {
        v6 = (char *)v8;
        v7 = 0xFFFFFFFFLL;
      }
      v8[0] = v7;
      result = *(_QWORD *)(v3 + 512);
      if ( (void (*)())result != nullsub_360 )
        return ((__int64 (__fastcall *)(__int64 *, char *, __int64))result)(a1, v6, 4);
    }
    else
    {
      return (__int64)v5(a1, 0xFFFFFFFF, 4u);
    }
  }
  return result;
}
