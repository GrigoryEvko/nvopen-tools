// Function: sub_2FB0DC0
// Address: 0x2fb0dc0
//
__int64 __fastcall sub_2FB0DC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v6; // rdx
  __int64 *v7; // rcx
  __int64 v8; // rax

  v6 = 16LL * *(unsigned int *)(a3 + 24);
  v7 = (__int64 *)(v6 + a1[1]);
  v8 = *v7;
  if ( (*v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v7[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v8 = sub_2FB0650(a1, a2, a3, (__int64)v7, a5);
    v6 = 16LL * *(unsigned int *)(a3 + 24);
  }
  if ( v8 == *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + v6 + 8) )
    return a3 + 48;
  else
    return *(_QWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 16);
}
