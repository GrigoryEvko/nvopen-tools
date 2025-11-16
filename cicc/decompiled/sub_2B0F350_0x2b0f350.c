// Function: sub_2B0F350
// Address: 0x2b0f350
//
__int64 __fastcall sub_2B0F350(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v3; // r13
  __int64 v4; // r8
  _DWORD *v5; // rbx
  __int64 v6; // r9
  __int64 v7; // rax
  int v8; // r15d
  unsigned __int64 v9; // rdx
  _QWORD *v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  int v13; // [rsp+18h] [rbp-38h]
  unsigned int v14; // [rsp+1Ch] [rbp-34h]

  v13 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*a2 - 64LL) + 8LL) + 32LL);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0xC00000000LL;
  v11 = &a2[a3];
  if ( v11 != a2 )
  {
    v3 = a2;
    v4 = 0;
    do
    {
      v5 = *(_DWORD **)(*v3 + 72LL);
      v6 = (__int64)&v5[*(unsigned int *)(*v3 + 80LL)];
      if ( (_DWORD *)v6 != v5 )
      {
        v7 = *(unsigned int *)(a1 + 8);
        do
        {
          v8 = *v5 + v4;
          v9 = v7 + 1;
          if ( *v5 == -1 )
            v8 = -1;
          if ( v9 > *(unsigned int *)(a1 + 12) )
          {
            v12 = v6;
            v14 = v4;
            sub_C8D5F0(a1, (const void *)(a1 + 16), v9, 4u, v4, v6);
            v7 = *(unsigned int *)(a1 + 8);
            v6 = v12;
            v4 = v14;
          }
          ++v5;
          *(_DWORD *)(*(_QWORD *)a1 + 4 * v7) = v8;
          v7 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v7;
        }
        while ( (_DWORD *)v6 != v5 );
      }
      v4 = (unsigned int)(v13 + v4);
      ++v3;
    }
    while ( v11 != v3 );
  }
  return a1;
}
