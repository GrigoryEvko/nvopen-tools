// Function: sub_B49600
// Address: 0xb49600
//
__int64 __fastcall sub_B49600(__int64 a1, const void *a2, size_t a3)
{
  __int64 v3; // rbp
  __int64 v5; // rdx
  _QWORD v7[2]; // [rsp-10h] [rbp-10h] BYREF

  v5 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v5 )
    return 0;
  v7[1] = v3;
  v7[0] = *(_QWORD *)(v5 + 120);
  return sub_A747B0(v7, -1, a2, a3);
}
