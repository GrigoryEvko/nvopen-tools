// Function: sub_CB6A20
// Address: 0xcb6a20
//
__int64 __fastcall sub_CB6A20(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // edx
  unsigned int v6; // r13d
  unsigned int v7; // esi
  void *v8; // rdi
  unsigned __int64 v9; // r14
  unsigned __int8 *v10; // rsi

  v4 = *(unsigned int *)(a2 + 16) - *(_QWORD *)(a2 + 8);
  if ( v4 <= 0 )
  {
    v6 = 0;
    v7 = 0;
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 20);
    if ( v5 == 2 )
    {
      v7 = v4;
      v6 = 0;
    }
    else if ( v5 == 3 )
    {
      v7 = v4 >> 1;
      v6 = v4 - v7;
    }
    else
    {
      v6 = 0;
      if ( v5 == 1 )
        v6 = v4;
      v7 = 0;
    }
  }
  sub_CB69B0(a1, v7);
  v8 = *(void **)(a1 + 32);
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(unsigned __int8 **)a2;
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v8 < v9 )
  {
    sub_CB6200(a1, v10, *(_QWORD *)(a2 + 8));
  }
  else if ( v9 )
  {
    memcpy(v8, v10, *(_QWORD *)(a2 + 8));
    *(_QWORD *)(a1 + 32) += v9;
  }
  sub_CB69B0(a1, v6);
  return a1;
}
