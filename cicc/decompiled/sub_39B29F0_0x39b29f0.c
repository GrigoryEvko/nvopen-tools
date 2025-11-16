// Function: sub_39B29F0
// Address: 0x39b29f0
//
__int64 __fastcall sub_39B29F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int8 a8,
        unsigned __int8 a9,
        unsigned __int8 a10)
{
  unsigned int v15; // edx
  __int64 v16; // r12
  void (__fastcall *v17)(__int64, __int64); // rax
  __int64 (__fastcall *v19)(__int64, __int64, __int64, __int64, _QWORD, _QWORD); // r11
  __int64 (__fastcall *v20)(__int64, __int64, __int64, __int64, __int64, __int64); // r11
  __int64 v21; // r9

  v15 = *(_DWORD *)(a2 + 52);
  if ( v15 == 3 )
  {
    v19 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(a1 + 152);
    if ( v19 )
      v16 = v19(a3, a4, a5, a6, a8, a10);
    else
      v16 = sub_39F24F0(a3, a4, a5, a6, a8, a10, 0);
  }
  else if ( v15 > 3 )
  {
    v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(a1 + 168);
    v21 = a8;
    if ( v20 )
    {
LABEL_11:
      v16 = v20(a2, a3, a4, a5, a6, v21);
      goto LABEL_5;
    }
    v16 = sub_39F4760(a3, a4, a5, a6, a8);
  }
  else
  {
    if ( v15 == 1 )
    {
      v16 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(a1 + 144))(
              a3,
              a4,
              a5,
              a6,
              a8,
              a9);
      goto LABEL_5;
    }
    v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(a1 + 160);
    v21 = a8;
    if ( v20 )
      goto LABEL_11;
    v16 = sub_39F0A90(a3, a4, a5, a6, a8);
  }
LABEL_5:
  v17 = *(void (__fastcall **)(__int64, __int64))(a1 + 192);
  if ( v17 )
    v17(v16, a7);
  return v16;
}
