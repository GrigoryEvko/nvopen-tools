// Function: sub_16E8650
// Address: 0x16e8650
//
__int64 __fastcall sub_16E8650(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v6; // rsi
  int v7; // ecx
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  void *v13; // rdi
  size_t v14; // rdx
  char *v15; // rsi
  size_t v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  int v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  char v21; // [rsp+28h] [rbp-38h]

  v3 = a2[5];
  v4 = a2[6];
  if ( v3 != v4 )
  {
    while ( 1 )
    {
      if ( !*(_DWORD *)v3 )
        goto LABEL_4;
      if ( *(_DWORD *)v3 == 2 )
      {
        v13 = *(void **)(a1 + 24);
        v14 = *(_QWORD *)(v3 + 16);
        v15 = *(char **)(v3 + 8);
        if ( *(_QWORD *)(a1 + 16) - (_QWORD)v13 < v14 )
        {
LABEL_13:
          sub_16E7EE0(a1, v15, v14);
          goto LABEL_4;
        }
LABEL_9:
        if ( !v14 )
          goto LABEL_4;
        v3 += 64;
        v17 = v14;
        memcpy(v13, v15, v14);
        *(_QWORD *)(a1 + 24) += v17;
        if ( v4 == v3 )
          return a1;
      }
      else
      {
        v11 = a2[2];
        v12 = *(_QWORD *)(v3 + 24);
        if ( v12 >= (a2[3] - v11) >> 3 )
        {
          v13 = *(void **)(a1 + 24);
          v14 = *(_QWORD *)(v3 + 16);
          v15 = *(char **)(v3 + 8);
          if ( v14 > *(_QWORD *)(a1 + 16) - (_QWORD)v13 )
            goto LABEL_13;
          goto LABEL_9;
        }
        v6 = *(_QWORD *)(v11 + 8 * v12);
        v7 = *(_DWORD *)(v3 + 40);
        v8 = *(_BYTE *)(v3 + 44);
        v20 = *(_QWORD *)(v3 + 32);
        v9 = *(_QWORD *)(v3 + 48);
        v19 = v7;
        v10 = *(_QWORD *)(v3 + 56);
        v18 = v6;
        v21 = v8;
        sub_16E8170(&v18, a1, v9, v10);
LABEL_4:
        v3 += 64;
        if ( v4 == v3 )
          return a1;
      }
    }
  }
  return a1;
}
