// Function: sub_33E4DA0
// Address: 0x33e4da0
//
void __fastcall sub_33E4DA0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdi
  __int64 v11; // rax

  if ( a4 )
  {
    v5 = a4;
    if ( a4 == 1 )
    {
      v11 = *a3;
      *(_DWORD *)(a2 + 104) = 1;
      *(_QWORD *)(a2 + 96) = v11 & 0xFFFFFFFFFFFFFFFBLL;
    }
    else
    {
      v6 = a1[24];
      v7 = 8 * a4;
      a1[34] += 8 * a4;
      v8 = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v9 = 8 * a4 + v8;
      if ( a1[25] >= v9 && v6 )
      {
        a1[24] = v9;
        v10 = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v10 = sub_9D1E70((__int64)(a1 + 24), v7, v7, 3);
        v8 = v10 & 0xFFFFFFFFFFFFFFFBLL;
      }
      if ( v7 )
        memmove((void *)v10, a3, v7);
      *(_DWORD *)(a2 + 104) = v5;
      *(_QWORD *)(a2 + 96) = v8 | 4;
    }
  }
  else
  {
    *(_QWORD *)(a2 + 96) = 0;
    *(_DWORD *)(a2 + 104) = 0;
  }
}
