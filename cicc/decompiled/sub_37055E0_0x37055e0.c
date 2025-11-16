// Function: sub_37055E0
// Address: 0x37055e0
//
unsigned __int64 *__fastcall sub_37055E0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned int v6; // ecx
  unsigned __int64 v7; // rax
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_370E660(v8, a2 + 16);
  if ( (v8[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v8[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    v8[0] = 0;
    sub_9C66B0(v8);
    v5 = *(_QWORD *)(a2 + 8);
    v6 = *(_DWORD *)(v5 + 56) - *(_DWORD *)(a2 + 104);
    *(_QWORD *)(v5 + 56) = *(unsigned int *)(a2 + 104);
    sub_1254950((unsigned __int64 *)v8, *(_QWORD *)(a2 + 8), a3 + 8, v6);
    v7 = v8[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v8[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v8[0] = 0;
      *a1 = v7 | 1;
    }
    else
    {
      v8[0] = 0;
      sub_9C66B0(v8);
      *a1 = 1;
    }
    sub_9C66B0(v8);
    return a1;
  }
}
