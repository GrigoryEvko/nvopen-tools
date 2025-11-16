// Function: sub_C0C3A0
// Address: 0xc0c3a0
//
__int64 __fastcall sub_C0C3A0(__int64 a1, char *a2, size_t a3)
{
  __int64 v3; // r13
  __int64 v4; // rcx
  int v5; // r9d
  int v6; // ebx
  size_t v8; // r12
  int v9; // r11d
  int v10; // r10d
  unsigned int i; // r12d
  __int64 v12; // r14
  int v13; // r8d
  unsigned int v14; // r12d
  const void *v16; // rsi
  bool v17; // al
  int v18; // eax
  size_t v19; // [rsp+0h] [rbp-50h]
  int v20; // [rsp+8h] [rbp-48h]
  int v21; // [rsp+Ch] [rbp-44h]
  int v22; // [rsp+10h] [rbp-40h]
  int v23; // [rsp+14h] [rbp-3Ch]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v3 )
  {
    v5 = v3 - 1;
    v6 = a3;
    v8 = HIDWORD(a3);
    v9 = 1;
    a3 = (unsigned int)a3;
    v10 = v8;
    for ( i = (v3 - 1) & v8; ; i = v5 & v14 )
    {
      v12 = v4 + 24LL * i;
      v13 = *(_DWORD *)(v12 + 12);
      if ( v10 == v13 )
      {
        v16 = *(const void **)v12;
        v17 = a2 + 1 == 0;
        if ( *(_QWORD *)v12 != -1 )
        {
          v17 = a2 + 2 == 0;
          if ( v16 != (const void *)-2LL )
          {
            if ( v6 != *(_DWORD *)(v12 + 8) )
              goto LABEL_4;
            v20 = v9;
            v21 = *(_DWORD *)(v12 + 12);
            v22 = v5;
            v23 = v10;
            v24 = v4;
            if ( !a3 )
              return *(_QWORD *)(v12 + 16);
            v19 = a3;
            v18 = memcmp(a2, v16, a3);
            v9 = v20;
            v13 = v21;
            v5 = v22;
            v10 = v23;
            v4 = v24;
            a3 = v19;
            v17 = v18 == 0;
          }
        }
        if ( v17 )
          return *(_QWORD *)(v12 + 16);
      }
LABEL_4:
      if ( !v13 && *(_QWORD *)v12 == -1 )
        break;
      v14 = v9 + i;
      ++v9;
    }
  }
  v12 = v4 + 24 * v3;
  return *(_QWORD *)(v12 + 16);
}
