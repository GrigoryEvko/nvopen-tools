// Function: sub_1E86160
// Address: 0x1e86160
//
__int64 __fastcall sub_1E86160(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // eax
  __int64 v7; // r15
  __int64 v8; // r8
  unsigned __int64 v10; // rdx
  __int64 v11; // r14
  unsigned int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // rcx
  _QWORD *v17; // rdx

  v6 = a2 & 0x7FFFFFFF;
  v7 = a2 & 0x7FFFFFFF;
  v8 = 8 * v7;
  v10 = *(unsigned int *)(a1 + 408);
  if ( (unsigned int)v10 <= (a2 & 0x7FFFFFFFu) || (v11 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8LL * v6)) == 0 )
  {
    v13 = v6 + 1;
    if ( (unsigned int)v10 < v6 + 1 )
    {
      if ( v13 < v10 )
      {
        *(_DWORD *)(a1 + 408) = v13;
      }
      else if ( v13 > v10 )
      {
        if ( v13 > (unsigned __int64)*(unsigned int *)(a1 + 412) )
        {
          sub_16CD150(a1 + 400, (const void *)(a1 + 416), v13, 8, 8 * a2, a6);
          v10 = *(unsigned int *)(a1 + 408);
          v8 = 8LL * (a2 & 0x7FFFFFFF);
        }
        v14 = *(_QWORD *)(a1 + 400);
        v15 = *(_QWORD *)(a1 + 416);
        v16 = (_QWORD *)(v14 + 8LL * v13);
        v17 = (_QWORD *)(v14 + 8 * v10);
        if ( v16 != v17 )
        {
          do
            *v17++ = v15;
          while ( v16 != v17 );
          v14 = *(_QWORD *)(a1 + 400);
        }
        *(_DWORD *)(a1 + 408) = v13;
        goto LABEL_6;
      }
    }
    v14 = *(_QWORD *)(a1 + 400);
LABEL_6:
    *(_QWORD *)(v14 + v8) = sub_1DBA290(a2);
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v7);
    sub_1DBB110((_QWORD *)a1, v11);
  }
  return v11;
}
