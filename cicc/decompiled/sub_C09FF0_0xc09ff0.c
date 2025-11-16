// Function: sub_C09FF0
// Address: 0xc09ff0
//
__int64 __fastcall sub_C09FF0(__int64 a1, _DWORD *a2, _DWORD *a3, _BYTE *a4, size_t a5)
{
  unsigned int v7; // r8d
  unsigned __int64 v8; // rcx
  _BYTE *v10; // rdi
  int v12; // eax
  __int64 v13; // rax
  char v14; // bl
  unsigned __int64 v16; // [rsp+8h] [rbp-58h]
  _QWORD v17[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = 1;
  v8 = *(_QWORD *)(a1 + 8);
  if ( v8 >= a5 )
  {
    v10 = *(_BYTE **)a1;
    if ( !a5 || (v16 = v8, v12 = memcmp(v10, a4, a5), v8 = v16, v7 = 1, !v12) )
    {
      *(_QWORD *)a1 = &v10[a5];
      *(_QWORD *)(a1 + 8) = v8 - a5;
      *a2 = sub_C09F10(a4, a5);
      v13 = *(_QWORD *)(a1 + 8);
      if ( v13 && **(_BYTE **)a1 == 110 )
      {
        v14 = 1;
        ++*(_QWORD *)a1;
        *(_QWORD *)(a1 + 8) = v13 - 1;
      }
      else
      {
        v14 = 0;
      }
      if ( (unsigned __int8)sub_C93C00(a1, 10, v17) || v17[0] != SLODWORD(v17[0]) )
        *a3 = 1;
      else
        *a3 = v17[0];
      v7 = 0;
      if ( v14 )
        *a3 = -*a3;
    }
  }
  return v7;
}
