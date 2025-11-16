// Function: sub_38EC000
// Address: 0x38ec000
//
__int64 __fastcall sub_38EC000(__int64 a1, __int64 a2)
{
  __int64 v3; // [rsp+10h] [rbp-40h] BYREF
  __int64 v4; // [rsp+18h] [rbp-38h] BYREF
  const char *v5; // [rsp+20h] [rbp-30h] BYREF
  char v6; // [rsp+30h] [rbp-20h]
  char v7; // [rsp+31h] [rbp-1Fh]

  v3 = 0;
  v4 = 0;
  if ( (unsigned __int8)sub_38EBF60(a1, &v3, a2) )
    return 1;
  v7 = 1;
  v6 = 3;
  v5 = "unexpected token in directive";
  if ( (unsigned __int8)sub_3909E20(a1, 25, &v5) || (unsigned __int8)sub_38EBF60(a1, &v4, a2) )
    return 1;
  (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 328) + 856LL))(*(_QWORD *)(a1 + 328), v3, v4);
  return 0;
}
