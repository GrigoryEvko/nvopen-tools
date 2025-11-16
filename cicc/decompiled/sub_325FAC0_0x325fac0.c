// Function: sub_325FAC0
// Address: 0x325fac0
//
__int64 __fastcall sub_325FAC0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  int v8; // eax
  __int64 result; // rax
  __int64 v11; // rax
  __int64 *v12; // rax

  v7 = a2;
  v8 = *(_DWORD *)(a2 + 24);
  if ( v8 == 186 )
  {
    if ( (unsigned __int8)sub_33E2390(
                            a1,
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                            1) )
    {
      v11 = *(_QWORD *)(a2 + 40);
      *(_QWORD *)a5 = *(_QWORD *)(v11 + 40);
      *(_DWORD *)(a5 + 8) = *(_DWORD *)(v11 + 48);
      v12 = *(__int64 **)(a2 + 40);
      v7 = *v12;
      a3 = *((_DWORD *)v12 + 2);
    }
    v8 = *(_DWORD *)(v7 + 24);
  }
  result = (v8 - 190) & 0xFFFFFFFD;
  if ( !(_DWORD)result )
  {
    *(_QWORD *)a4 = v7;
    *(_DWORD *)(a4 + 8) = a3;
  }
  return result;
}
