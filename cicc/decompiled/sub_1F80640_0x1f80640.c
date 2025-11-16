// Function: sub_1F80640
// Address: 0x1f80640
//
__int64 __fastcall sub_1F80640(__int64 a1)
{
  char *v1; // rdx
  char v2; // al
  __int64 v3; // rdx
  _BYTE v5[8]; // [rsp+0h] [rbp-20h] BYREF
  __int64 v6; // [rsp+8h] [rbp-18h]

  v1 = *(char **)(a1 + 40);
  v2 = *v1;
  v3 = *((_QWORD *)v1 + 1);
  v5[0] = v2;
  v6 = v3;
  if ( !v2 )
    sub_1F58D30((__int64)v5);
  return *(_QWORD *)(a1 + 88);
}
