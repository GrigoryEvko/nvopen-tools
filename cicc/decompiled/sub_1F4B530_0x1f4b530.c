// Function: sub_1F4B530
// Address: 0x1f4b530
//
__int64 __fastcall sub_1F4B530(__int64 a1, signed int a2, __int64 a3)
{
  unsigned __int64 v5; // r8
  bool v6; // dl
  unsigned __int8 v7; // di
  char v8; // al
  char v9; // di
  __int64 result; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  int v14; // edx

  if ( a2 > 0 )
  {
    v11 = (unsigned __int64)sub_1F4ABE0(a1, a2, 1);
    return *(unsigned int *)(*(_QWORD *)(a1 + 280)
                           + 24LL
                           * (*(unsigned __int16 *)(*(_QWORD *)v11 + 24LL)
                            + *(_DWORD *)(a1 + 288)
                            * (unsigned int)((__int64)(*(_QWORD *)(a1 + 264) - *(_QWORD *)(a1 + 256)) >> 3)));
  }
  if ( a2 && (v12 = a2 & 0x7FFFFFFF, (unsigned int)v12 < *(_DWORD *)(a3 + 336)) )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(a3 + 328) + 8 * v12);
    v7 = v13 & 1;
    v5 = v13 >> 2;
    v6 = (v13 & 2) != 0;
  }
  else
  {
    v5 = 0;
    v6 = 0;
    v7 = 0;
  }
  v8 = (4 * v5) | v7 | (2 * v6);
  if ( !((4 * v5) | (unsigned __int16)(v7 | (unsigned __int16)(2 * v6)) & 0xFFFC) )
    goto LABEL_9;
  v9 = v7 & 1;
  if ( (v8 & 2) != 0 )
  {
    v14 = v9 ? WORD1(v5) : v5 >> 16;
    result = v14 * (unsigned int)(unsigned __int16)v5;
  }
  else
  {
    result = (unsigned int)v5;
    if ( v9 )
      result = (unsigned __int16)v5;
  }
  if ( !(_DWORD)result )
  {
LABEL_9:
    v11 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    return *(unsigned int *)(*(_QWORD *)(a1 + 280)
                           + 24LL
                           * (*(unsigned __int16 *)(*(_QWORD *)v11 + 24LL)
                            + *(_DWORD *)(a1 + 288)
                            * (unsigned int)((__int64)(*(_QWORD *)(a1 + 264) - *(_QWORD *)(a1 + 256)) >> 3)));
  }
  return result;
}
