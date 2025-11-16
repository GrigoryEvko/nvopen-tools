// Function: sub_1EC18F0
// Address: 0x1ec18f0
//
__int64 __fastcall sub_1EC18F0(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v6; // r14
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r15
  unsigned int v10; // ebx
  __int64 v11; // rcx
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // [rsp-48h] [rbp-48h]
  __int64 v16; // [rsp-40h] [rbp-40h]

  result = a2 & 0x7FFFFFFF;
  v3 = result;
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 264LL) + 4 * result) )
  {
    v4 = 8 * result;
    v6 = *(_QWORD *)(a1 + 264);
    v7 = *(unsigned int *)(v6 + 408);
    if ( (unsigned int)result < (unsigned int)v7 )
    {
      v8 = *(_QWORD *)(v6 + 400);
      v9 = *(_QWORD *)(v8 + 8 * result);
      if ( v9 )
        goto LABEL_4;
    }
    v10 = result + 1;
    if ( (unsigned int)v7 < (int)result + 1 )
    {
      if ( v10 >= v7 )
      {
        if ( v10 > v7 )
        {
          if ( v10 > (unsigned __int64)*(unsigned int *)(v6 + 412) )
          {
            v15 = 8 * result;
            sub_16CD150(v6 + 400, (const void *)(v6 + 416), v10, 8, result, v4);
            v7 = *(unsigned int *)(v6 + 408);
            v4 = v15;
            v3 = a2 & 0x7FFFFFFF;
          }
          v11 = *(_QWORD *)(v6 + 400);
          v12 = (_QWORD *)(v11 + 8 * v7);
          v13 = (_QWORD *)(v11 + 8LL * v10);
          v14 = *(_QWORD *)(v6 + 416);
          if ( v13 != v12 )
          {
            do
              *v12++ = v14;
            while ( v13 != v12 );
            v11 = *(_QWORD *)(v6 + 400);
          }
          *(_DWORD *)(v6 + 408) = v10;
          goto LABEL_7;
        }
      }
      else
      {
        *(_DWORD *)(v6 + 408) = v10;
      }
    }
    v11 = *(_QWORD *)(v6 + 400);
LABEL_7:
    v16 = v3;
    *(_QWORD *)(v11 + v4) = sub_1DBA290(a2);
    v9 = *(_QWORD *)(*(_QWORD *)(v6 + 400) + 8 * v16);
    sub_1DBB110((_QWORD *)v6, v9);
LABEL_4:
    sub_21031A0(*(_QWORD *)(a1 + 272), v9, v7, v8, v3);
    return sub_1EC1590(a1, (char **)(a1 + 880), v9);
  }
  return result;
}
