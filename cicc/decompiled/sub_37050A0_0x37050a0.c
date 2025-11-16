// Function: sub_37050A0
// Address: 0x37050a0
//
unsigned __int64 *__fastcall sub_37050A0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned int v7; // ecx
  unsigned __int64 v8; // rax
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_37141A0(v9, a2 + 16, a3, a4);
  if ( (v9[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v9[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    v9[0] = 0;
    sub_9C66B0(v9);
    v6 = *(_QWORD *)(a2 + 8);
    v7 = *(_DWORD *)(v6 + 56) - *(_DWORD *)(a2 + 104);
    *(_QWORD *)(v6 + 56) = *(unsigned int *)(a2 + 104);
    sub_1254950((unsigned __int64 *)v9, *(_QWORD *)(a2 + 8), a3 + 8, v7);
    v8 = v9[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v9[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v9[0] = 0;
      *a1 = v8 | 1;
    }
    else
    {
      v9[0] = 0;
      sub_9C66B0(v9);
      *a1 = 1;
    }
    sub_9C66B0(v9);
    return a1;
  }
}
