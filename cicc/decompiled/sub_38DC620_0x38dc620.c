// Function: sub_38DC620
// Address: 0x38dc620
//
_QWORD *__fastcall sub_38DC620(
        __int64 a1,
        unsigned int a2,
        int a3,
        int a4,
        unsigned __int16 a5,
        char a6,
        char a7,
        int a8,
        int a9,
        unsigned __int64 a10)
{
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rax
  char v18; // r9
  __int64 v20; // rdi
  const char *v21; // rax
  const char *v24; // [rsp+10h] [rbp-50h] BYREF
  char v25; // [rsp+20h] [rbp-40h]
  char v26; // [rsp+21h] [rbp-3Fh]

  v13 = sub_38BE350(*(_QWORD *)(a1 + 8));
  v14 = sub_390FE40(v13, a2);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 16);
    v16 = v14;
    v17 = *(unsigned int *)(a1 + 120);
    if ( !v15 )
    {
      if ( (_DWORD)v17 )
        v15 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v17 - 32);
      *(_QWORD *)(v16 + 16) = v15;
      goto LABEL_5;
    }
    if ( (_DWORD)v17 && v15 == *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v17 - 32) )
    {
LABEL_5:
      v18 = *(_BYTE *)(v13 + 14);
      *(_DWORD *)v13 = a2;
      *(_DWORD *)(v13 + 4) = a3;
      *(_DWORD *)(v13 + 8) = a4;
      *(_WORD *)(v13 + 12) = a5;
      *(_BYTE *)(v13 + 16) = 1;
      *(_BYTE *)(v13 + 14) = v18 & 0xFC | (a6 | (2 * a7)) & 3;
      return (_QWORD *)a5;
    }
    v26 = 1;
    v20 = *(_QWORD *)(a1 + 8);
    v21 = "all .cv_loc directives for a function must be in the same section";
  }
  else
  {
    v26 = 1;
    v20 = *(_QWORD *)(a1 + 8);
    v21 = "function id not introduced by .cv_func_id or .cv_inline_site_id";
  }
  v24 = v21;
  v25 = 3;
  return sub_38BE3D0(v20, a10, (__int64)&v24);
}
