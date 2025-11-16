// Function: sub_25282F0
// Address: 0x25282f0
//
__int64 __fastcall sub_25282F0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned int v4; // r13d
  int v5; // edx
  __int64 v7; // rdx
  int v8; // ecx
  int v9; // eax
  unsigned int v10; // eax
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  __int64 v15; // [rsp+8h] [rbp-28h]

  v4 = 1;
  v5 = *a2;
  if ( (unsigned int)(v5 - 12) <= 1 )
    return v4;
  if ( (_BYTE)v5 == 60 )
  {
    if ( ((*(_DWORD *)(*(_QWORD *)(a1 + 208) + 376LL) - 26) & 0xFFFFFFEE) != 0 )
    {
      v11 = sub_250D2C0((unsigned __int64)a2, 0);
      v15 = v12;
      v14 = v11;
      v4 = sub_2553E90(a1, &v14, 89, 0);
      if ( !(_BYTE)v4 )
      {
        v13 = sub_2527F10(a1, v14, v15, a3, 1, 0, 1);
        if ( v13 )
          LOBYTE(v4) = (*(_WORD *)(v13 + 98) & 7) == 7;
      }
    }
    return v4;
  }
  if ( (_BYTE)v5 == 3 )
  {
    v4 = a2[80] & 1;
    if ( (a2[80] & 1) != 0 )
      return v4;
    v4 = 1;
    if ( (a2[33] & 0x1C) != 0 )
      return v4;
  }
  if ( ((*(_DWORD *)(*(_QWORD *)(a1 + 208) + 376LL) - 26) & 0xFFFFFFEE) != 0 )
    return 0;
  v7 = *((_QWORD *)a2 + 1);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) > 1 )
  {
    v4 = 1;
    if ( *(_DWORD *)(v7 + 8) >> 8 != 5 )
      goto LABEL_13;
    return v4;
  }
  v4 = 1;
  v9 = *(_DWORD *)(**(_QWORD **)(v7 + 16) + 8LL) >> 8;
  if ( v9 == 5 )
    return v4;
  if ( v8 != 18 )
  {
LABEL_13:
    if ( v8 == 17 )
      v10 = *(_DWORD *)(**(_QWORD **)(v7 + 16) + 8LL);
    else
      v10 = *(_DWORD *)(v7 + 8);
    v9 = v10 >> 8;
  }
  LOBYTE(v4) = v9 == 4;
  return v4;
}
