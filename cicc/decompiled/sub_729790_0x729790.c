// Function: sub_729790
// Address: 0x729790
//
__int64 *__fastcall sub_729790(int a1, __int64 a2, int a3)
{
  __int64 v4; // rbx
  int v5; // r14d
  __int64 v6; // rbx
  __int64 *result; // rax
  __int64 **v8; // rcx
  __int64 *v9; // rdx
  __int64 v10; // rcx

  v4 = (int)sub_8230C0();
  v5 = v4;
  v6 = unk_4F072B8 + 16 * v4;
  if ( a3 )
  {
    *(_DWORD *)(v6 + 8) = a3;
  }
  else
  {
    *(_DWORD *)(v6 + 8) = sub_823220();
    *(_BYTE *)(a2 + 208) |= 1u;
    a3 = *(_DWORD *)(v6 + 8);
  }
  sub_7296B0(a3);
  result = (__int64 *)sub_726EB0(17, a1, a2);
  *(_QWORD *)v6 = result;
  *(_DWORD *)(a2 + 160) = v5;
  *(_DWORD *)(a2 + 164) = *(_DWORD *)(v6 + 8);
  v8 = (__int64 **)(unk_4F072B0 + 8LL * dword_4F07270[0]);
  v9 = *v8;
  if ( *v8 )
  {
    v10 = *v9;
    if ( *v9 )
    {
      *(_QWORD *)(v10 + 8) = result;
      *result = v10;
    }
    *v9 = (__int64)result;
    result[1] = (__int64)v9;
  }
  else
  {
    *v8 = result;
  }
  return result;
}
