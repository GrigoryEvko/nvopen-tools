// Function: sub_2673950
// Address: 0x2673950
//
unsigned __int64 __fastcall sub_2673950(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // r13d
  unsigned __int8 *v4; // r14
  unsigned __int8 *v5; // r12
  unsigned __int64 result; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx

  v2 = *(_QWORD *)(a2 + 208);
  v3 = *(_DWORD *)(v2 + 72LL * *(int *)(a1 + 100) + 34584);
  v4 = *(unsigned __int8 **)(v2 + 160LL * *(int *)(v2 + 72LL * *(int *)(a1 + 100) + 34644) + 3632);
  v5 = (unsigned __int8 *)(*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL);
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v5 = (unsigned __int8 *)*((_QWORD *)v5 + 3);
  result = *v5;
  if ( (unsigned __int8)result > 0x1Cu )
  {
    v7 = (unsigned int)(result - 34);
    if ( (unsigned __int8)v7 > 0x33u || (v8 = 0x8000000000041LL, !_bittest64(&v8, v7)) )
    {
      result = sub_B43CB0((__int64)v5);
      v5 = (unsigned __int8 *)result;
      goto LABEL_5;
    }
    result = sub_250C680((__int64 *)(a1 + 72));
    if ( result )
    {
      v5 = *(unsigned __int8 **)(result + 24);
      goto LABEL_5;
    }
    result = (unsigned __int64)sub_BD3990(*((unsigned __int8 **)v5 - 4), a2);
    v5 = (unsigned __int8 *)result;
    if ( result && !*(_BYTE *)result )
      goto LABEL_5;
  }
  else
  {
    if ( !(_BYTE)result )
      goto LABEL_5;
    if ( (_BYTE)result == 22 )
    {
      v5 = (unsigned __int8 *)*((_QWORD *)v5 + 3);
      goto LABEL_5;
    }
  }
  v5 = 0;
LABEL_5:
  if ( v4 == v5 )
  {
    *(_DWORD *)(a1 + 104) = v3;
  }
  else
  {
    result = *(unsigned __int8 *)(a1 + 96);
    *(_BYTE *)(a1 + 97) = result;
  }
  return result;
}
