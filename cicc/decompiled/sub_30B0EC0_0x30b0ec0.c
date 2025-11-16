// Function: sub_30B0EC0
// Address: 0x30b0ec0
//
__int64 __fastcall sub_30B0EC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rdi
  _BYTE *v8; // rax

  v2 = a1;
  v3 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == v3 )
  {
    a1 = sub_CB6200(a1, (unsigned __int8 *)"[", 1u);
  }
  else
  {
    *v3 = 91;
    ++*(_QWORD *)(a1 + 32);
  }
  v4 = sub_30B0E00(a1, *(_DWORD *)(a2 + 8));
  v5 = *(_QWORD *)(v4 + 32);
  v6 = v4;
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v5) <= 4 )
  {
    v6 = sub_CB6200(v4, "] to ", 5u);
  }
  else
  {
    *(_DWORD *)v5 = 1869881437;
    *(_BYTE *)(v5 + 4) = 32;
    *(_QWORD *)(v4 + 32) += 5LL;
  }
  v7 = sub_CB5A80(v6, *(_QWORD *)a2);
  v8 = *(_BYTE **)(v7 + 32);
  if ( *(_BYTE **)(v7 + 24) == v8 )
  {
    sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
    return v2;
  }
  else
  {
    *v8 = 10;
    ++*(_QWORD *)(v7 + 32);
    return v2;
  }
}
