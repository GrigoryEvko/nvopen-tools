// Function: sub_392DFB0
// Address: 0x392dfb0
//
__int64 __fastcall sub_392DFB0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // rax
  const char *v8; // rax
  __int64 result; // rax
  unsigned __int64 v10; // rdx
  const char *v11; // [rsp+0h] [rbp-20h] BYREF
  char v12; // [rsp+10h] [rbp-10h]
  char v13; // [rsp+11h] [rbp-Fh]

  v7 = *(_QWORD *)(a4 + 160);
  if ( v7 > 3 && *(_DWORD *)(*(_QWORD *)(a4 + 152) + v7 - 4) == 1870095406 )
  {
    v13 = 1;
    v8 = "A dwo section may not contain relocations";
LABEL_4:
    v11 = v8;
    v12 = 3;
    sub_38BE3D0(a2, a3, (__int64)&v11);
    return 0;
  }
  result = 1;
  if ( a5 )
  {
    v10 = *(_QWORD *)(a5 + 160);
    if ( v10 > 3 && *(_DWORD *)(*(_QWORD *)(a5 + 152) + v10 - 4) == 1870095406 )
    {
      v13 = 1;
      v8 = "A relocation may not refer to a dwo section";
      goto LABEL_4;
    }
  }
  return result;
}
