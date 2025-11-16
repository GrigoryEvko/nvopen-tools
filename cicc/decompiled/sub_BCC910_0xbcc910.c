// Function: sub_BCC910
// Address: 0xbcc910
//
__int64 __fastcall sub_BCC910(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r12d
  __int64 v6; // r14
  int v7; // r12d
  unsigned int v8; // r15d
  _QWORD *v9; // rcx
  __int64 v10; // rdx
  int v11; // r8d
  _QWORD *v12; // r9
  __int64 v13; // rax
  size_t v14; // rax
  int v15; // eax
  _QWORD *v16; // [rsp+8h] [rbp-58h]
  _QWORD *v17; // [rsp+10h] [rbp-50h]
  int v18; // [rsp+1Ch] [rbp-44h]
  unsigned __int64 v19[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v4 - 1;
    v19[0] = sub_BCC330(*(_QWORD **)a2, *(_QWORD *)a2 + 8LL * *(_QWORD *)(a2 + 8));
    v8 = v7 & sub_BCC160((__int64 *)v19, (_BYTE *)(a2 + 16));
    v9 = (_QWORD *)(v6 + 8LL * v8);
    v10 = *v9;
    if ( *v9 != -4096 )
    {
      v11 = 1;
      v12 = 0;
      while ( 1 )
      {
        if ( v10 == -8192 )
        {
          if ( !v12 )
            v12 = v9;
        }
        else if ( ((*(_DWORD *)(v10 + 8) & 0x200) != 0) == *(_BYTE *)(a2 + 16) )
        {
          v13 = *(_QWORD *)(a2 + 8);
          if ( v13 == *(_DWORD *)(v10 + 12) )
          {
            v14 = 8 * v13;
            v17 = v12;
            v18 = v11;
            if ( !v14
              || (v16 = v9,
                  v15 = memcmp(*(const void **)a2, *(const void **)(v10 + 16), v14),
                  v9 = v16,
                  v11 = v18,
                  v12 = v17,
                  !v15) )
            {
              *a3 = v9;
              return 1;
            }
          }
        }
        v8 = v7 & (v11 + v8);
        v9 = (_QWORD *)(v6 + 8LL * v8);
        v10 = *v9;
        if ( *v9 == -4096 )
          break;
        ++v11;
      }
      if ( v12 )
        v9 = v12;
    }
    *a3 = v9;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
