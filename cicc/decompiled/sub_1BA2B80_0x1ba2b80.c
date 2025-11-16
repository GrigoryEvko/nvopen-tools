// Function: sub_1BA2B80
// Address: 0x1ba2b80
//
__int64 __fastcall sub_1BA2B80(__int64 a1, __int64 a2, int a3)
{
  int v3; // r14d
  __int64 v5; // r12
  char v6; // r8
  __int64 v7; // rax
  unsigned int v8; // esi
  int v9; // ebx
  int v10; // edx
  __int64 v11; // rdx
  __int64 v12; // [rsp+18h] [rbp-48h] BYREF
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  int v14; // [rsp+28h] [rbp-38h]

  if ( a3 == 1 )
  {
    sub_1B8DFF0(a2);
    v3 = sub_14A3620(*(_QWORD *)(a1 + 328));
    return v3 + (unsigned int)sub_14A34A0(*(_QWORD *)(a1 + 328));
  }
  else
  {
    v5 = a1 + 264;
    v13 = a2;
    v14 = a3;
    v6 = sub_1B99450(a1 + 264, &v13, &v12);
    v7 = v12;
    if ( v6 )
    {
      return *(unsigned int *)(v12 + 20);
    }
    else
    {
      v8 = *(_DWORD *)(a1 + 288);
      v9 = *(_DWORD *)(a1 + 280);
      ++*(_QWORD *)(a1 + 264);
      v10 = v9 + 1;
      if ( 4 * (v9 + 1) >= 3 * v8 )
      {
        sub_1BA2A10(v5, 2 * v8);
        sub_1B99450(v5, &v13, &v12);
        v7 = v12;
        v10 = *(_DWORD *)(a1 + 280) + 1;
      }
      else if ( v8 - *(_DWORD *)(a1 + 284) - v10 <= v8 >> 3 )
      {
        sub_1BA2A10(v5, v8);
        sub_1B99450(v5, &v13, &v12);
        v7 = v12;
        v10 = *(_DWORD *)(a1 + 280) + 1;
      }
      *(_DWORD *)(a1 + 280) = v10;
      if ( *(_QWORD *)v7 != -8 || *(_DWORD *)(v7 + 8) != -1 )
        --*(_DWORD *)(a1 + 284);
      v11 = v13;
      *(_QWORD *)(v7 + 16) = 0;
      *(_QWORD *)v7 = v11;
      *(_DWORD *)(v7 + 8) = v14;
      return 0;
    }
  }
}
