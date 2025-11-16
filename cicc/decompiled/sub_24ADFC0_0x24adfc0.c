// Function: sub_24ADFC0
// Address: 0x24adfc0
//
void __fastcall sub_24ADFC0(_QWORD *a1, __int64 a2)
{
  _DWORD *v2; // rax
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r9
  __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 *v8; // rdx
  __int64 v9; // r8
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r8
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  int v15; // r11d
  unsigned __int64 v16; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v17; // [rsp+8h] [rbp-18h]

  v2 = (_DWORD *)a1[2];
  v16 = *(_QWORD *)(*(_QWORD *)(a1[6] + 448LL) + 8LL * (unsigned int)(*v2)++);
  v3 = a1[6];
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v3 + 280);
  v6 = *(unsigned int *)(v3 + 296);
  if ( !(_DWORD)v6 )
  {
LABEL_13:
    v10 = v16;
LABEL_14:
    v11 = v10;
    goto LABEL_15;
  }
  v7 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( v4 != *v8 )
  {
    v15 = 1;
    while ( v9 != -4096 )
    {
      v7 = (v6 - 1) & (v15 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( v4 == *v8 )
        goto LABEL_3;
      ++v15;
    }
    goto LABEL_13;
  }
LABEL_3:
  v10 = v16;
  v11 = v16;
  if ( v8 == (__int64 *)(v5 + 16 * v6) )
    goto LABEL_14;
  v12 = v8[1];
  if ( !v12 )
    goto LABEL_14;
  v13 = *(_QWORD *)(v12 + 16);
  if ( v13 < v16 )
  {
    *(_QWORD *)(v12 + 16) = v16;
    *(_BYTE *)(v12 + 24) = 1;
  }
  if ( v13 > v10 )
  {
    v14 = v13 - v10;
    v17 = v14;
    if ( v14 > v10 )
    {
      v11 = v14;
    }
    else if ( !v10 )
    {
      return;
    }
LABEL_18:
    sub_24AD4F0(*(_QWORD *)(*a1 + 40LL), a2, &v16, 2, v11);
    return;
  }
LABEL_15:
  v17 = 0;
  if ( v10 )
    goto LABEL_18;
}
