// Function: sub_31C2CB0
// Address: 0x31c2cb0
//
__int64 __fastcall sub_31C2CB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int8 *v7; // rax
  __int64 v8; // rax
  int v9; // r14d
  __int64 v10; // rbx
  __int64 *v11; // rdi

  v4 = a3;
  v5 = *(_QWORD *)(a3 + 24);
  v6 = sub_318B6A0(a3);
  v7 = sub_98ACB0(*(unsigned __int8 **)(v6 + 16), 6u);
  v8 = sub_3188190(v5, (__int64)v7);
  v9 = *(_DWORD *)(v4 + 32);
  v10 = v8;
  if ( sub_318B630(v4) && (*(_DWORD *)(v4 + 8) != 37 || sub_318B6C0(v4)) )
  {
    if ( sub_318B670(v4) )
    {
      v4 = sub_318B680(v4);
    }
    else if ( *(_DWORD *)(v4 + 8) == 37 )
    {
      v4 = sub_318B6C0(v4);
    }
  }
  v11 = sub_318EB80(v4);
  if ( (unsigned int)*(unsigned __int8 *)(*v11 + 8) - 17 <= 1 )
    v11 = sub_318E560(v11);
  *(_DWORD *)a1 = v9;
  *(_QWORD *)(a1 + 16) = v10;
  *(_QWORD *)(a1 + 8) = v11;
  return a1;
}
