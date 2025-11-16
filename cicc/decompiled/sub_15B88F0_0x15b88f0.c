// Function: sub_15B88F0
// Address: 0x15b88f0
//
__int64 __fastcall sub_15B88F0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  int v9; // ebx
  __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rsi
  int v14; // r8d
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // [rsp+0h] [rbp-40h] BYREF
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  char v21; // [rsp+10h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = v4 - 1;
    v10 = *(unsigned int *)(*a2 + 8);
    v19 = *(_QWORD *)(*a2 + 8 * (1 - v10));
    v11 = *(_QWORD *)(v6 + 8 * (2 - v10));
    LOBYTE(v6) = *(_BYTE *)(v6 + 24) & 1;
    v20 = v11;
    v21 = v6;
    v12 = sub_15B2420(&v19, &v20);
    v13 = *a2;
    v14 = 1;
    v15 = 0;
    v16 = v9 & v12;
    v17 = (_QWORD *)(v7 + 8LL * v16);
    v18 = *v17;
    if ( *v17 == *a2 )
    {
LABEL_10:
      *a3 = v17;
      return 1;
    }
    else
    {
      while ( v18 != -8 )
      {
        if ( v18 != -16 || v15 )
          v17 = v15;
        v16 = v9 & (v14 + v16);
        v18 = *(_QWORD *)(v7 + 8LL * v16);
        if ( v18 == v13 )
        {
          v17 = (_QWORD *)(v7 + 8LL * v16);
          goto LABEL_10;
        }
        ++v14;
        v15 = v17;
        v17 = (_QWORD *)(v7 + 8LL * v16);
      }
      if ( !v15 )
        v15 = v17;
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
