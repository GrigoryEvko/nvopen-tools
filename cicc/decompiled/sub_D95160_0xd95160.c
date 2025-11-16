// Function: sub_D95160
// Address: 0xd95160
//
__int64 __fastcall sub_D95160(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rdx
  unsigned __int64 v6; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-18h]

  v2 = *(_DWORD *)(a2 + 8);
  v7 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43780((__int64)&v6, (const void **)a2);
    v2 = v7;
    if ( v7 > 0x40 )
    {
      sub_C43D10((__int64)&v6);
      v2 = v7;
      v4 = v6;
      goto LABEL_5;
    }
    v3 = v6;
  }
  else
  {
    v3 = *(_QWORD *)a2;
  }
  v4 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & ~v3;
  if ( !v2 )
    v4 = 0;
LABEL_5:
  *(_DWORD *)(a1 + 8) = v2;
  *(_QWORD *)a1 = v4;
  return a1;
}
