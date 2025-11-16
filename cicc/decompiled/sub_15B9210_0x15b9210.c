// Function: sub_15B9210
// Address: 0x15b9210
//
__int64 __fastcall sub_15B9210(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // r14
  int v8; // ebx
  int v9; // eax
  __int64 v10; // rsi
  int v11; // r8d
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  _QWORD *v14; // rcx
  __int64 v15; // rdx

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v8 = v4 - 1;
    v9 = sub_15B1DB0(*(__int64 **)(*a2 + 24), *(_QWORD *)(*a2 + 32));
    v10 = *a2;
    v11 = 1;
    v12 = 0;
    v13 = v8 & v9;
    v14 = (_QWORD *)(v6 + 8LL * v13);
    v15 = *v14;
    if ( *v14 == *a2 )
    {
LABEL_10:
      *a3 = v14;
      return 1;
    }
    else
    {
      while ( v15 != -8 )
      {
        if ( v15 != -16 || v12 )
          v14 = v12;
        v13 = v8 & (v11 + v13);
        v15 = *(_QWORD *)(v6 + 8LL * v13);
        if ( v15 == v10 )
        {
          v14 = (_QWORD *)(v6 + 8LL * v13);
          goto LABEL_10;
        }
        ++v11;
        v12 = v14;
        v14 = (_QWORD *)(v6 + 8LL * v13);
      }
      if ( !v12 )
        v12 = v14;
      *a3 = v12;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
