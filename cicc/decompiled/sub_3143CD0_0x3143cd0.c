// Function: sub_3143CD0
// Address: 0x3143cd0
//
__int64 __fastcall sub_3143CD0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        unsigned __int8 *a4,
        __int64 a5,
        __int64 *a6,
        _DWORD *a7)
{
  __int64 result; // rax
  bool v11; // zf
  __int64 *v12; // rax
  __int64 v13; // r8
  unsigned __int8 v14; // al
  __int64 v15; // rsi
  int v16; // ecx
  unsigned int v17; // esi
  int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 *v21; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v22; // [rsp+8h] [rbp-28h] BYREF

  if ( !(unsigned __int8)sub_3143680(a1, a3, a4) )
    return 0;
  v11 = (unsigned __int8)sub_3140DC0(a5, a6, &v21) == 0;
  v12 = v21;
  if ( v11 )
  {
    v16 = *(_DWORD *)(a5 + 16);
    v17 = *(_DWORD *)(a5 + 24);
    v22 = v21;
    ++*(_QWORD *)a5;
    v18 = v16 + 1;
    if ( 4 * (v16 + 1) >= 3 * v17 )
    {
      v17 *= 2;
    }
    else if ( v17 - *(_DWORD *)(a5 + 20) - v18 > v17 >> 3 )
    {
      goto LABEL_11;
    }
    sub_3141720(a5, v17);
    sub_3140DC0(a5, a6, &v22);
    v18 = *(_DWORD *)(a5 + 16) + 1;
    v12 = v22;
LABEL_11:
    *(_DWORD *)(a5 + 16) = v18;
    if ( v12[2] != -4096 || v12[1] != -4096 || *v12 != -4096 )
      --*(_DWORD *)(a5 + 20);
    v12[2] = a6[2];
    v12[1] = a6[1];
    v19 = *a6;
    v12[3] = 0;
    *v12 = v19;
  }
  v13 = v12[3];
  v14 = *(_BYTE *)(a2 - 16);
  if ( (v14 & 2) != 0 )
  {
    if ( *(_DWORD *)(a2 - 24) != 2 )
    {
LABEL_6:
      v15 = 0;
      goto LABEL_7;
    }
    v20 = *(_QWORD *)(a2 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_6;
    v20 = a2 - 16 - 8LL * ((v14 >> 2) & 0xF);
  }
  v15 = *(_QWORD *)(v20 + 8);
LABEL_7:
  result = sub_3140D50(a1, v15, v13);
  if ( !(_BYTE)result )
    return 0;
  ++*a7;
  return result;
}
