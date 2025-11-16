// Function: sub_C47CA0
// Address: 0xc47ca0
//
__int64 __fastcall sub_C47CA0(__int64 a1, __int64 a2, __int64 a3, bool *a4)
{
  unsigned int v6; // r14d
  unsigned __int64 v8; // [rsp+0h] [rbp-40h]
  unsigned int v9; // [rsp+Ch] [rbp-34h]

  v9 = *(_DWORD *)(a3 + 8);
  v6 = *(_DWORD *)(a2 + 8);
  if ( v9 > 0x40 )
  {
    v8 = *(unsigned int *)(a2 + 8);
    if ( v9 - (unsigned int)sub_C444A0(a3) <= 0x40 && v8 >= **(_QWORD **)a3 )
      v6 = **(_QWORD **)a3;
  }
  else if ( (unsigned __int64)*(unsigned int *)(a2 + 8) >= *(_QWORD *)a3 )
  {
    v6 = *(_QWORD *)a3;
  }
  sub_C47B80(a1, a2, v6, a4);
  return a1;
}
