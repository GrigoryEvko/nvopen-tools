// Function: sub_D92430
// Address: 0xd92430
//
__int64 __fastcall sub_D92430(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // ebx
  int v5; // eax
  unsigned int v6; // ebx
  int v7; // eax
  int v8; // eax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned int v11; // [rsp+4h] [rbp-2Ch]
  __int64 v12; // [rsp+10h] [rbp-20h] BYREF
  int v13; // [rsp+18h] [rbp-18h]

  if ( !*(_BYTE *)(a2 + 16) )
  {
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v4 = *(_DWORD *)(a2 + 8);
  if ( a3 > 1 && a3 < v4 )
  {
    if ( v4 <= 0x40 )
    {
      if ( !*(_QWORD *)a2 )
        goto LABEL_8;
      _BitScanReverse64(&v10, *(_QWORD *)a2);
      v6 = 64 - (v10 ^ 0x3F);
    }
    else
    {
      v11 = a3;
      v5 = sub_C444A0(a2);
      a3 = v11;
      v6 = v4 - v5;
    }
    if ( a3 >= v6 )
    {
LABEL_8:
      sub_C44740((__int64)&v12, (char **)a2, a3);
      v7 = v13;
      *(_BYTE *)(a1 + 16) = 1;
      *(_DWORD *)(a1 + 8) = v7;
      *(_QWORD *)a1 = v12;
      return a1;
    }
  }
  *(_BYTE *)(a1 + 16) = 0;
  v8 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a1 + 8) = v8;
  v9 = *(_QWORD *)a2;
  *(_BYTE *)(a1 + 16) = 1;
  *(_QWORD *)a1 = v9;
  return a1;
}
