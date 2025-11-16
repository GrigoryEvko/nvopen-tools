// Function: sub_B9FB90
// Address: 0xb9fb90
//
__int64 __fastcall sub_B9FB90(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // dl
  __int64 *v4; // rcx
  __int64 *v5; // rdx
  __int64 result; // rax
  unsigned int v7; // esi
  int v8; // eax
  __int64 *v9; // rdx
  int v10; // eax
  __int64 v11[2]; // [rsp+8h] [rbp-68h] BYREF
  __int64 *v12; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v13[4]; // [rsp+20h] [rbp-50h] BYREF
  int v14; // [rsp+40h] [rbp-30h]

  v11[0] = a1;
  v13[0] = 0;
  v13[1] = 0;
  v3 = *(_BYTE *)(a1 - 16);
  if ( (v3 & 2) != 0 )
  {
    v4 = *(__int64 **)(a1 - 32);
    v5 = &v4[*(unsigned int *)(a1 - 24)];
  }
  else
  {
    v4 = (__int64 *)(a1 - 16 - 8LL * ((v3 >> 2) & 0xF));
    v5 = &v4[(*(_WORD *)(a1 - 16) >> 6) & 0xF];
  }
  v13[2] = v4;
  v13[3] = (__int64 *)(v5 - v4);
  v14 = *(_DWORD *)(a1 + 4);
  result = sub_B903E0(a2, (__int64)v13);
  if ( !result )
  {
    if ( (unsigned __int8)sub_B95B20(a2, v11, &v12) )
      return v11[0];
    v7 = *(_DWORD *)(a2 + 24);
    v8 = *(_DWORD *)(a2 + 16);
    v9 = v12;
    ++*(_QWORD *)a2;
    v10 = v8 + 1;
    v13[0] = v9;
    if ( 4 * v10 >= 3 * v7 )
    {
      v7 *= 2;
    }
    else if ( v7 - *(_DWORD *)(a2 + 20) - v10 > v7 >> 3 )
    {
LABEL_10:
      *(_DWORD *)(a2 + 16) = v10;
      if ( *v9 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v9 = v11[0];
      return v11[0];
    }
    sub_B9C570(a2, v7);
    sub_B95B20(a2, v11, v13);
    v9 = v13[0];
    v10 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_10;
  }
  return result;
}
