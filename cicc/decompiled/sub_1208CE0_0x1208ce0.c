// Function: sub_1208CE0
// Address: 0x1208ce0
//
__int64 __fastcall sub_1208CE0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v3; // eax
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  int v6; // [rsp+8h] [rbp-18h]

  if ( *(_BYTE *)(a2 + 12) )
  {
    sub_C44AB0((__int64)&v5, a2, a3);
    v3 = v6;
    *(_BYTE *)(a1 + 12) = 1;
  }
  else
  {
    sub_C44B10((__int64)&v5, (char **)a2, a3);
    v3 = v6;
    *(_BYTE *)(a1 + 12) = 0;
  }
  *(_DWORD *)(a1 + 8) = v3;
  *(_QWORD *)a1 = v5;
  return a1;
}
