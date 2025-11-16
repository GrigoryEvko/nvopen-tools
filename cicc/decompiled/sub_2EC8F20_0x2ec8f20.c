// Function: sub_2EC8F20
// Address: 0x2ec8f20
//
_QWORD *__fastcall sub_2EC8F20(__int64 a1, __int64 a2)
{
  bool v2; // zf
  _QWORD *result; // rax
  __int64 v4; // r8
  __int64 v5; // r8
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v2 = (*(_DWORD *)(a1 + 24) & *(_DWORD *)(a2 + 204)) == 0;
  v6 = a2;
  if ( v2 )
  {
    result = sub_2EC1410(*(_QWORD **)(a1 + 128), *(_QWORD *)(a1 + 136), &v6);
    *(_DWORD *)(*result + 204LL) &= ~*(_DWORD *)(v5 + 88);
    *result = *(_QWORD *)(*(_QWORD *)(v5 + 136) - 8LL);
    *(_QWORD *)(v5 + 136) -= 8LL;
  }
  else
  {
    result = sub_2EC1410(*(_QWORD **)(a1 + 64), *(_QWORD *)(a1 + 72), &v6);
    *(_DWORD *)(*result + 204LL) &= ~*(_DWORD *)(v4 + 24);
    *result = *(_QWORD *)(*(_QWORD *)(v4 + 72) - 8LL);
    *(_QWORD *)(v4 + 72) -= 8LL;
  }
  return result;
}
