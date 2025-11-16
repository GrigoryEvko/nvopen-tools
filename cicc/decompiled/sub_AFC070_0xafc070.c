// Function: sub_AFC070
// Address: 0xafc070
//
__int64 __fastcall sub_AFC070(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // r14
  int v8; // ebx
  int v9; // eax
  __int64 v10; // rsi
  unsigned int v11; // eax
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  int v14; // r8d
  _QWORD *v15; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v8 = v4 - 1;
    v9 = sub_AF6940(*(__int64 **)(*a2 + 136), *(_QWORD *)(*a2 + 136) + 8LL * *(unsigned int *)(*a2 + 144));
    v10 = *a2;
    v11 = v8 & v9;
    v12 = (_QWORD *)(v6 + 8LL * v11);
    v13 = *v12;
    if ( *a2 == *v12 )
    {
LABEL_4:
      *a3 = v12;
      return 1;
    }
    else
    {
      v14 = 1;
      v15 = 0;
      while ( v13 != -4096 )
      {
        if ( v13 == -8192 && !v15 )
          v15 = v12;
        v11 = v8 & (v14 + v11);
        v12 = (_QWORD *)(v6 + 8LL * v11);
        v13 = *v12;
        if ( *v12 == v10 )
          goto LABEL_4;
        ++v14;
      }
      if ( !v15 )
        v15 = v12;
      *a3 = v15;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
