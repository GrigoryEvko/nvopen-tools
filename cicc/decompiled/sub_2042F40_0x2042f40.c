// Function: sub_2042F40
// Address: 0x2042f40
//
unsigned __int64 __fastcall sub_2042F40(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  int v3; // ecx
  unsigned int *v4; // rcx
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  unsigned __int64 result; // rax
  int v10; // eax
  _BYTE *v11; // rsi
  _BYTE *v12; // rsi
  __int64 *v13; // [rsp+8h] [rbp-18h] BYREF

  v13 = a2;
  if ( !(unsigned __int8)sub_20421A0((_QWORD *)a1, a2)
    || (v2 = *v13, (v3 = *(_DWORD *)(*v13 + 56)) != 0)
    && (v4 = (unsigned int *)(*(_QWORD *)(v2 + 32) + 40LL * (unsigned int)(v3 - 1)),
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v4 + 40LL) + 16LL * v4[2]) == 111) )
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 160) + 8LL) = 0;
    v5 = *(_QWORD *)(a1 + 168);
    if ( *(_QWORD *)(a1 + 176) != v5 )
      *(_QWORD *)(a1 + 176) = v5;
    v2 = *a2;
    if ( !*a2 )
      goto LABEL_8;
  }
  v6 = *(__int16 *)(v2 + 24);
  if ( (v6 & 0x8000u) == 0 )
  {
LABEL_8:
    *(_DWORD *)(*(_QWORD *)(a1 + 160) + 8LL) = 0;
    v7 = *(_QWORD *)(a1 + 168);
    if ( v7 != *(_QWORD *)(a1 + 176) )
      *(_QWORD *)(a1 + 176) = v7;
    v8 = 0;
    goto LABEL_11;
  }
  v10 = ~v6;
  if ( v10 <= 10 )
  {
    if ( v10 > 6 )
      goto LABEL_18;
    goto LABEL_17;
  }
  if ( v10 != 14 )
LABEL_17:
    sub_20E8BC0(*(_QWORD *)(a1 + 160), *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL) + ((__int64)v10 << 6));
LABEL_18:
  v11 = *(_BYTE **)(a1 + 176);
  if ( v11 == *(_BYTE **)(a1 + 184) )
  {
    sub_1CFD630(a1 + 168, v11, &v13);
    v8 = (__int64)(*(_QWORD *)(a1 + 176) - *(_QWORD *)(a1 + 168)) >> 3;
  }
  else
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = v13;
      v11 = *(_BYTE **)(a1 + 176);
    }
    v12 = v11 + 8;
    *(_QWORD *)(a1 + 176) = v12;
    v8 = (__int64)&v12[-*(_QWORD *)(a1 + 168)] >> 3;
  }
LABEL_11:
  result = **(unsigned int **)(a1 + 152);
  if ( result <= v8 )
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 160) + 8LL) = 0;
    result = *(_QWORD *)(a1 + 168);
    if ( result != *(_QWORD *)(a1 + 176) )
      *(_QWORD *)(a1 + 176) = result;
  }
  return result;
}
