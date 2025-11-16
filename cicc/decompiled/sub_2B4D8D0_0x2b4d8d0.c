// Function: sub_2B4D8D0
// Address: 0x2b4d8d0
//
__int64 __fastcall sub_2B4D8D0(__int64 a1, const void **a2, _QWORD *a3)
{
  int v4; // r12d
  __int64 v6; // r14
  int v7; // r12d
  int v8; // eax
  __int64 v9; // r8
  int v10; // r11d
  __int64 v11; // r10
  unsigned int v12; // r15d
  size_t v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  int v16; // eax
  unsigned int v17; // r15d
  size_t v18; // [rsp+0h] [rbp-A0h]
  __int64 v19; // [rsp+10h] [rbp-90h]
  __int64 v20; // [rsp+18h] [rbp-88h]
  int v21; // [rsp+24h] [rbp-7Ch]
  __int64 v22; // [rsp+28h] [rbp-78h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v4 - 1;
  v8 = sub_939680(*a2, (__int64)*a2 + 4 * *((unsigned int *)a2 + 2));
  v9 = *((unsigned int *)a2 + 2);
  v10 = 1;
  v11 = 0;
  v12 = v7 & v8;
  v13 = 4 * v9;
  while ( 1 )
  {
    v14 = v6 + 40LL * v12;
    v15 = *(unsigned int *)(v14 + 8);
    if ( v9 == v15 )
    {
      v19 = *(unsigned int *)(v14 + 8);
      v20 = v9;
      v21 = v10;
      v22 = v11;
      if ( !v13 )
        goto LABEL_12;
      v18 = v13;
      v16 = memcmp(*a2, *(const void **)v14, v13);
      v14 = v6 + 40LL * v12;
      v13 = v18;
      v11 = v22;
      v10 = v21;
      v9 = v20;
      v15 = v19;
      if ( !v16 )
      {
LABEL_12:
        *a3 = v14;
        return 1;
      }
    }
    if ( v15 == 1 )
      break;
LABEL_16:
    v17 = v10 + v12;
    ++v10;
    v12 = v7 & v17;
  }
  if ( **(_DWORD **)v14 != -2 )
  {
    if ( **(_DWORD **)v14 == -3 && !v11 )
      v11 = v14;
    goto LABEL_16;
  }
  if ( !v11 )
    v11 = v14;
  *a3 = v11;
  return 0;
}
