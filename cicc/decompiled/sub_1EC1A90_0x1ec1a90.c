// Function: sub_1EC1A90
// Address: 0x1ec1a90
//
__int64 __fastcall sub_1EC1A90(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v10; // r8
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r13
  unsigned int v14; // r13d
  __int64 v15; // rcx
  __int64 v16; // rax
  _QWORD *v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rdx

  result = a2 & 0x7FFFFFFF;
  v7 = result;
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 416) + 264LL) + 4 * result) )
  {
    v8 = *(_QWORD *)(a1 - 408);
    v10 = 8 * result;
    v11 = *(unsigned int *)(v8 + 408);
    if ( (unsigned int)result < (unsigned int)v11 )
    {
      v12 = *(_QWORD *)(v8 + 400);
      v13 = *(_QWORD *)(v12 + 8 * result);
      if ( v13 )
        goto LABEL_4;
    }
    v14 = result + 1;
    if ( (unsigned int)v11 < (int)result + 1 )
    {
      v16 = v14;
      if ( v14 >= v11 )
      {
        if ( v14 > v11 )
        {
          if ( v14 > (unsigned __int64)*(unsigned int *)(v8 + 412) )
          {
            sub_16CD150(v8 + 400, (const void *)(v8 + 416), v14, 8, v10, a6);
            v11 = *(unsigned int *)(v8 + 408);
            v10 = 8 * v7;
            v16 = v14;
          }
          v15 = *(_QWORD *)(v8 + 400);
          v17 = (_QWORD *)(v15 + 8 * v16);
          v18 = (_QWORD *)(v15 + 8 * v11);
          v19 = *(_QWORD *)(v8 + 416);
          if ( v17 != v18 )
          {
            do
              *v18++ = v19;
            while ( v17 != v18 );
            v15 = *(_QWORD *)(v8 + 400);
          }
          *(_DWORD *)(v8 + 408) = v14;
          goto LABEL_7;
        }
      }
      else
      {
        *(_DWORD *)(v8 + 408) = v14;
      }
    }
    v15 = *(_QWORD *)(v8 + 400);
LABEL_7:
    *(_QWORD *)(v15 + v10) = sub_1DBA290(a2);
    v13 = *(_QWORD *)(*(_QWORD *)(v8 + 400) + 8 * v7);
    sub_1DBB110((_QWORD *)v8, v13);
LABEL_4:
    sub_21031A0(*(_QWORD *)(a1 - 400), v13, v11, v12, v10);
    return sub_1EC1590(a1 - 672, (char **)(a1 + 208), v13);
  }
  return result;
}
