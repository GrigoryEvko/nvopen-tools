// Function: sub_A744F0
// Address: 0xa744f0
//
__int64 __fastcall sub_A744F0(__int64 a1, _QWORD *a2, int a3)
{
  __int64 v3; // rbx
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v7; // eax
  __int64 v8; // [rsp+0h] [rbp-20h] BYREF
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v9[0] = sub_A744E0(a2, a3);
  v8 = sub_A734C0(v9, 97);
  if ( v8 )
  {
    v3 = sub_A72AA0(&v8);
    v4 = *(_DWORD *)(v3 + 8);
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
    {
      sub_C43780(a1, v3);
      v7 = *(_DWORD *)(v3 + 24);
      *(_DWORD *)(a1 + 24) = v7;
      if ( v7 <= 0x40 )
        goto LABEL_4;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)v3;
      v5 = *(_DWORD *)(v3 + 24);
      *(_DWORD *)(a1 + 24) = v5;
      if ( v5 <= 0x40 )
      {
LABEL_4:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(v3 + 16);
LABEL_5:
        *(_BYTE *)(a1 + 32) = 1;
        return a1;
      }
    }
    sub_C43780(a1 + 16, v3 + 16);
    goto LABEL_5;
  }
  *(_BYTE *)(a1 + 32) = 0;
  return a1;
}
