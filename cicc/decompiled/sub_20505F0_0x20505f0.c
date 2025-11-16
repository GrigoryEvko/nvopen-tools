// Function: sub_20505F0
// Address: 0x20505f0
//
__int64 __fastcall sub_20505F0(__int64 a1, __int64 a2)
{
  _DWORD *v2; // r10
  char *v3; // r13
  _DWORD *v4; // r14
  unsigned int v5; // ebx
  unsigned int v6; // eax
  int v7; // r9d
  unsigned int v8; // r11d
  __int64 v9; // rdx
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  _DWORD *v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+18h] [rbp-48h]
  unsigned __int64 v15; // [rsp+20h] [rbp-40h]
  unsigned int v16; // [rsp+2Ch] [rbp-34h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v2 = *(_DWORD **)(a2 + 136);
  v3 = *(char **)(a2 + 80);
  v13 = &v2[*(unsigned int *)(a2 + 144)];
  if ( v13 != v2 )
  {
    v4 = *(_DWORD **)(a2 + 136);
    v5 = 0;
    do
    {
      v6 = sub_2045180(*v3);
      v8 = v5 + *v4;
      if ( v8 != v5 )
      {
        v9 = *(unsigned int *)(a1 + 8);
        v10 = v6;
        do
        {
          v11 = *(unsigned int *)(*(_QWORD *)(a2 + 104) + 4LL * v5) | (unsigned __int64)(v10 << 32);
          if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v9 )
          {
            v14 = v10;
            v15 = *(unsigned int *)(*(_QWORD *)(a2 + 104) + 4LL * v5) | (unsigned __int64)(v10 << 32);
            v16 = v8;
            sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v10, v7);
            v9 = *(unsigned int *)(a1 + 8);
            v10 = v14;
            v11 = v15;
            v8 = v16;
          }
          ++v5;
          *(_QWORD *)(*(_QWORD *)a1 + 8 * v9) = v11;
          v9 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v9;
        }
        while ( v8 != v5 );
      }
      ++v4;
      ++v3;
    }
    while ( v13 != v4 );
  }
  return a1;
}
