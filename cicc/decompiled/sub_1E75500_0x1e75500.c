// Function: sub_1E75500
// Address: 0x1e75500
//
__int64 __fastcall sub_1E75500(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r12
  _BYTE *v4; // rsi
  __int64 result; // rax
  _BYTE *v6; // rsi
  _QWORD v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = a2;
  if ( *(_DWORD *)(a1 + 172) > a3 )
    *(_DWORD *)(a1 + 172) = a3;
  if ( !*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) && *(_DWORD *)(a1 + 164) < a3
    || (unsigned __int8)sub_1E72C10(a1, a2)
    || (v6 = *(_BYTE **)(a1 + 72), dword_4FC7CA0 <= (unsigned int)((__int64)&v6[-*(_QWORD *)(a1 + 64)] >> 3)) )
  {
    v7[0] = v3;
    v4 = *(_BYTE **)(a1 + 136);
    if ( v4 == *(_BYTE **)(a1 + 144) )
    {
      sub_1CFD630(a1 + 128, v4, v7);
      v3 = v7[0];
    }
    else
    {
      if ( v4 )
      {
        *(_QWORD *)v4 = v3;
        v4 = *(_BYTE **)(a1 + 136);
      }
      *(_QWORD *)(a1 + 136) = v4 + 8;
    }
    result = *(unsigned int *)(a1 + 88);
    *(_DWORD *)(v3 + 196) |= result;
  }
  else
  {
    v7[0] = v3;
    if ( v6 == *(_BYTE **)(a1 + 80) )
    {
      sub_1CFD630(a1 + 64, v6, v7);
      v3 = v7[0];
    }
    else
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = v3;
        v6 = *(_BYTE **)(a1 + 72);
      }
      *(_QWORD *)(a1 + 72) = v6 + 8;
    }
    result = *(unsigned int *)(a1 + 24);
    *(_DWORD *)(v3 + 196) |= result;
  }
  return result;
}
