// Function: sub_1AEA030
// Address: 0x1aea030
//
__int64 *__fastcall sub_1AEA030(__int64 *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 i; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // r9
  __int64 v10; // rax
  int v11; // r8d
  __int64 v12; // r14
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // r10
  __int64 v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  unsigned __int64 v19; // [rsp+0h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 23) & 0x10) != 0
    && (v3 = sub_161E8E0(a2)) != 0
    && (v4 = (__int64 *)sub_16498A0(a2), (v5 = sub_1629050(v4, v3)) != 0) )
  {
    v6 = *(_QWORD *)(v5 + 8);
    for ( i = 0; v6; v6 = *(_QWORD *)(v6 + 8) )
    {
      v17 = sub_1648700(v6);
      v18 = v17;
      if ( *((_BYTE *)v17 + 16) == 78 )
      {
        v8 = *(v17 - 3);
        if ( !*(_BYTE *)(v8 + 16) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v8 + 36) - 35) <= 2 )
        {
          v9 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (i & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (i & 4) != 0 )
            {
              v16 = *(unsigned int *)(v9 + 8);
              v13 = i & 0xFFFFFFFFFFFFFFF8LL;
              v15 = i & 0xFFFFFFFFFFFFFFF8LL;
            }
            else
            {
              v10 = sub_22077B0(48);
              v9 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( v10 )
              {
                *(_QWORD *)v10 = v10 + 16;
                *(_QWORD *)(v10 + 8) = 0x400000000LL;
              }
              v12 = v10;
              v13 = v10 & 0xFFFFFFFFFFFFFFF8LL;
              v14 = *(unsigned int *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 8);
              i = v12 | 4;
              v15 = v13;
              if ( (unsigned int)v14 >= *(_DWORD *)(v13 + 12) )
              {
                v19 = v9;
                sub_16CD150(v13, (const void *)(v13 + 16), 0, 8, v11, v9);
                v14 = *(unsigned int *)(v13 + 8);
                v9 = v19;
                v15 = v13;
              }
              *(_QWORD *)(*(_QWORD *)v13 + 8 * v14) = v9;
              v16 = (unsigned int)(*(_DWORD *)(v13 + 8) + 1);
              *(_DWORD *)(v13 + 8) = v16;
            }
            if ( *(_DWORD *)(v13 + 12) <= (unsigned int)v16 )
            {
              sub_16CD150(v15, (const void *)(v13 + 16), 0, 8, v11, v9);
              v16 = *(unsigned int *)(v13 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v13 + 8 * v16) = v18;
            ++*(_DWORD *)(v13 + 8);
          }
          else
          {
            i = (__int64)v18;
          }
        }
      }
    }
    *a1 = i;
  }
  else
  {
    *a1 = 0;
  }
  return a1;
}
