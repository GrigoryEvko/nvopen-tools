// Function: sub_BCC6E0
// Address: 0xbcc6e0
//
__int64 __fastcall sub_BCC6E0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r12d
  __int64 v6; // r15
  int v8; // eax
  int v9; // ecx
  unsigned int v10; // r14d
  _QWORD *v11; // r12
  __int64 v12; // rax
  int v13; // r8d
  _QWORD *v14; // r9
  _QWORD *v15; // rdx
  __int64 v16; // rsi
  int v17; // eax
  _QWORD *v18; // [rsp+0h] [rbp-50h]
  int v19; // [rsp+8h] [rbp-48h]
  int v20; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v21[0] = sub_BCC330(*(_QWORD **)(a2 + 8), *(_QWORD *)(a2 + 8) + 8LL * *(_QWORD *)(a2 + 16));
    v8 = sub_BCC1E0((__int64 *)a2, (__int64 *)v21, (_BYTE *)(a2 + 24));
    v9 = v4 - 1;
    v10 = (v4 - 1) & v8;
    v11 = (_QWORD *)(v6 + 8LL * v10);
    v12 = *v11;
    if ( *v11 != -4096 )
    {
      v13 = 1;
      v14 = 0;
      while ( 1 )
      {
        if ( v12 == -8192 )
        {
          if ( !v14 )
            v14 = v11;
        }
        else
        {
          v15 = *(_QWORD **)(v12 + 16);
          if ( *(_QWORD *)a2 == *v15 && *(_BYTE *)(a2 + 24) == (*(_DWORD *)(v12 + 8) >> 8 != 0) )
          {
            v16 = *(_QWORD *)(a2 + 16);
            if ( v16 == (8LL * *(unsigned int *)(v12 + 12) - 8) >> 3 )
            {
              v19 = v13;
              v18 = v14;
              v20 = v9;
              if ( !(8 * v16)
                || (v17 = memcmp(*(const void **)(a2 + 8), v15 + 1, 8 * v16), v9 = v20, v14 = v18, v13 = v19, !v17) )
              {
                *a3 = v11;
                return 1;
              }
            }
          }
        }
        v10 = v9 & (v13 + v10);
        v11 = (_QWORD *)(v6 + 8LL * v10);
        v12 = *v11;
        if ( *v11 == -4096 )
          break;
        ++v13;
      }
      if ( v14 )
        v11 = v14;
    }
    *a3 = v11;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
