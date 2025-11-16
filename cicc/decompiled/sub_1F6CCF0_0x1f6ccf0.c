// Function: sub_1F6CCF0
// Address: 0x1f6ccf0
//
__int64 __fastcall sub_1F6CCF0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  int v8; // eax
  __int64 result; // rax
  __int64 v11; // rax
  __int64 *v12; // rax

  v7 = a2;
  v8 = *(unsigned __int16 *)(a2 + 24);
  if ( (_WORD)v8 == 118 )
  {
    if ( sub_1D23600(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL)) )
    {
      v11 = *(_QWORD *)(a2 + 32);
      *(_QWORD *)a5 = *(_QWORD *)(v11 + 40);
      *(_DWORD *)(a5 + 8) = *(_DWORD *)(v11 + 48);
      v12 = *(__int64 **)(a2 + 32);
      v7 = *v12;
      a3 = *((_DWORD *)v12 + 2);
    }
    v8 = *(unsigned __int16 *)(v7 + 24);
  }
  result = (unsigned int)(v8 - 122);
  if ( (result & 0xFFFD) == 0 )
  {
    *(_QWORD *)a4 = v7;
    *(_DWORD *)(a4 + 8) = a3;
  }
  return result;
}
