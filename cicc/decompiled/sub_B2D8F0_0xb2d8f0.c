// Function: sub_B2D8F0
// Address: 0xb2d8f0
//
__int64 __fastcall sub_B2D8F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // eax
  unsigned int v4; // eax
  unsigned int v6; // eax
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v7[0] = sub_B2D8E0(a2, 97);
  if ( v7[0] )
  {
    v2 = sub_A72AA0(v7);
    v3 = *(_DWORD *)(v2 + 8);
    *(_DWORD *)(a1 + 8) = v3;
    if ( v3 > 0x40 )
    {
      sub_C43780(a1, v2);
      v6 = *(_DWORD *)(v2 + 24);
      *(_DWORD *)(a1 + 24) = v6;
      if ( v6 <= 0x40 )
        goto LABEL_4;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)v2;
      v4 = *(_DWORD *)(v2 + 24);
      *(_DWORD *)(a1 + 24) = v4;
      if ( v4 <= 0x40 )
      {
LABEL_4:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(v2 + 16);
LABEL_5:
        *(_BYTE *)(a1 + 32) = 1;
        return a1;
      }
    }
    sub_C43780(a1 + 16, v2 + 16);
    goto LABEL_5;
  }
  *(_BYTE *)(a1 + 32) = 0;
  return a1;
}
