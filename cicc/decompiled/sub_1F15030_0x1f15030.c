// Function: sub_1F15030
// Address: 0x1f15030
//
__int64 __fastcall sub_1F15030(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v7; // r14d
  unsigned int v8; // eax
  __int64 v9; // rdx
  int v10; // ecx
  __int64 v11; // r13
  __int64 v12; // r15
  unsigned __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 *v15; // rax
  unsigned int v16; // r8d
  unsigned int v18; // ebx
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rcx

  v7 = *(_DWORD *)(a1[5] + 112LL);
  v8 = v7 & 0x7FFFFFFF;
  v9 = v7 & 0x7FFFFFFF;
  v10 = *(_DWORD *)(*(_QWORD *)(a1[1] + 312LL) + 4 * v9);
  if ( v10 )
  {
    v7 = *(_DWORD *)(*(_QWORD *)(a1[1] + 312LL) + 4 * v9);
    v8 = v10 & 0x7FFFFFFF;
    v9 = v10 & 0x7FFFFFFF;
  }
  v11 = a1[2];
  v12 = 8 * v9;
  v13 = *(unsigned int *)(v11 + 408);
  if ( (unsigned int)v13 <= v8 || (v14 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + 8 * v9)) == 0 )
  {
    v18 = v8 + 1;
    if ( (unsigned int)v13 < v8 + 1 )
    {
      v20 = v18;
      if ( v18 < v13 )
      {
        *(_DWORD *)(v11 + 408) = v18;
      }
      else if ( v18 > v13 )
      {
        if ( v18 > (unsigned __int64)*(unsigned int *)(v11 + 412) )
        {
          sub_16CD150(v11 + 400, (const void *)(v11 + 416), v18, 8, a5, a6);
          v13 = *(unsigned int *)(v11 + 408);
          v20 = v18;
        }
        v19 = *(_QWORD *)(v11 + 400);
        v21 = (_QWORD *)(v19 + 8 * v20);
        v22 = (_QWORD *)(v19 + 8 * v13);
        v23 = *(_QWORD *)(v11 + 416);
        if ( v21 != v22 )
        {
          do
            *v22++ = v23;
          while ( v21 != v22 );
          v19 = *(_QWORD *)(v11 + 400);
        }
        *(_DWORD *)(v11 + 408) = v18;
        goto LABEL_13;
      }
    }
    v19 = *(_QWORD *)(v11 + 400);
LABEL_13:
    *(_QWORD *)(v19 + v12) = sub_1DBA290(v7);
    v14 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + v12);
    sub_1DBB110((_QWORD *)v11, v14);
  }
  v15 = (__int64 *)sub_1DB3C70((__int64 *)v14, a2);
  if ( v15 == (__int64 *)(*(_QWORD *)v14 + 24LL * *(unsigned int *)(v14 + 8))
    || (v16 = a2 & 0xFFFFFFF8,
        (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(a2 >> 1) & 3)) )
  {
    v16 = 0;
    if ( v15 != *(__int64 **)v14 )
      LOBYTE(v16) = *(v15 - 2) == a2;
  }
  else
  {
    LOBYTE(v16) = *v15 == a2;
  }
  return v16;
}
