// Function: sub_17EA5C0
// Address: 0x17ea5c0
//
void __fastcall sub_17EA5C0(_QWORD *a1, __int64 a2)
{
  _DWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // rcx
  int v7; // r11d
  unsigned int v8; // ebx
  __int64 *v9; // rdx
  __int64 v10; // r9
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r9
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v17; // [rsp+8h] [rbp-18h]

  v2 = (_DWORD *)a1[2];
  v16 = *(_QWORD *)(*(_QWORD *)(a1[6] + 344LL) + 8LL * (unsigned int)(*v2)++);
  v3 = a1[6];
  v4 = *(unsigned int *)(v3 + 296);
  if ( !(_DWORD)v4 )
  {
LABEL_12:
    v12 = v16;
LABEL_13:
    v11 = v12;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(v3 + 280);
  v7 = 1;
  v8 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( v5 != *v9 )
  {
    while ( v10 != -8 )
    {
      v8 = (v4 - 1) & (v7 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( v5 == *v9 )
        goto LABEL_3;
      ++v7;
    }
    goto LABEL_12;
  }
LABEL_3:
  v11 = v16;
  v12 = v16;
  if ( v9 == (__int64 *)(v6 + 16 * v4) )
    goto LABEL_13;
  v13 = v9[1];
  if ( !v13 )
    goto LABEL_13;
  v14 = *(_QWORD *)(v13 + 16);
  if ( v14 > v16 )
  {
    v15 = v14 - v16;
    v17 = v15;
    if ( v15 > v16 )
    {
      v11 = v15;
    }
    else if ( !v16 )
    {
      return;
    }
LABEL_17:
    sub_17E9890(*(__int64 **)(*a1 + 40LL), a2, &v16, 2, v11, v12);
    return;
  }
LABEL_14:
  v17 = 0;
  if ( v12 )
    goto LABEL_17;
}
