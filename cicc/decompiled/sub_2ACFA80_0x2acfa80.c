// Function: sub_2ACFA80
// Address: 0x2acfa80
//
__int64 __fastcall sub_2ACFA80(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rax
  int v6; // ecx
  unsigned int v7; // esi
  int v8; // edx
  char v9; // dl
  __int64 v10; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_2AC16F0(a1, (int *)a2, &v10) == 0;
  v4 = v10;
  if ( !v3 )
    return v10 + 8;
  v6 = *(_DWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v11[0] = v10;
  ++*(_QWORD *)a1;
  v8 = v6 + 1;
  if ( 4 * (v6 + 1) >= 3 * v7 )
  {
    v7 *= 2;
  }
  else if ( v7 - *(_DWORD *)(a1 + 20) - v8 > v7 >> 3 )
  {
    goto LABEL_5;
  }
  sub_2ACF7C0(a1, v7);
  sub_2AC16F0(a1, (int *)a2, v11);
  v8 = *(_DWORD *)(a1 + 16) + 1;
  v4 = v11[0];
LABEL_5:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *(_DWORD *)v4 != -1 || !*(_BYTE *)(v4 + 4) )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)v4 = *(_DWORD *)a2;
  v9 = *(_BYTE *)(a2 + 4);
  *(_QWORD *)(v4 + 8) = 0;
  *(_BYTE *)(v4 + 4) = v9;
  *(_QWORD *)(v4 + 16) = 0;
  *(_QWORD *)(v4 + 24) = 0;
  *(_DWORD *)(v4 + 32) = 0;
  return v4 + 8;
}
