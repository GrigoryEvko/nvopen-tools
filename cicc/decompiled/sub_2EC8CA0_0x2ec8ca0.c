// Function: sub_2EC8CA0
// Address: 0x2ec8ca0
//
__int64 __fastcall sub_2EC8CA0(__int64 a1)
{
  unsigned int v1; // r13d
  unsigned int v2; // ebx
  unsigned int v3; // eax
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 180);
  v2 = sub_2EC8BE0(a1, *(_QWORD **)(a1 + 64), (__int64)(*(_QWORD *)(a1 + 72) - *(_QWORD *)(a1 + 64)) >> 3);
  v3 = sub_2EC8BE0(a1, *(_QWORD **)(a1 + 128), (__int64)(*(_QWORD *)(a1 + 136) - *(_QWORD *)(a1 + 128)) >> 3);
  if ( v2 < v3 )
    v2 = v3;
  result = v1;
  if ( v2 >= v1 )
    return v2;
  return result;
}
