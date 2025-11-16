// Function: sub_22C23A0
// Address: 0x22c23a0
//
__int64 __fastcall sub_22C23A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // edx
  __int64 v6; // rbx
  __int64 v7; // rsi
  int v9; // r8d
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  char v13; // al
  __int64 v14; // rbx
  unsigned int v15; // esi
  int v16; // eax
  int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-68h] BYREF
  __int64 v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  void *v21; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v22[2]; // [rsp+28h] [rbp-48h] BYREF
  __int64 v23; // [rsp+38h] [rbp-38h]
  char v24; // [rsp+40h] [rbp-30h]
  unsigned __int64 v25[5]; // [rsp+48h] [rbp-28h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = v4 + 48LL * v5;
    v7 = *(_QWORD *)(v6 + 24);
    if ( v7 == a2 )
    {
LABEL_3:
      if ( v6 != v4 + 48 * v3 )
        return *(_QWORD *)(v6 + 40);
    }
    else
    {
      v9 = 1;
      while ( v7 != -4096 )
      {
        v5 = (v3 - 1) & (v9 + v5);
        v6 = v4 + 48LL * v5;
        v7 = *(_QWORD *)(v6 + 24);
        if ( v7 == a2 )
          goto LABEL_3;
        ++v9;
      }
    }
  }
  v10 = sub_22077B0(0x1C8u);
  if ( v10 )
  {
    memset((void *)v10, 0, 0x1C8u);
    *(_BYTE *)(v10 + 8) = 1;
    v11 = (_QWORD *)(v10 + 16);
    do
    {
      if ( v11 )
      {
        *v11 = 0;
        v11[1] = 0;
        v11[2] = -4096;
      }
      v11 += 8;
    }
    while ( v11 != (_QWORD *)(v10 + 272) );
    v21 = 0;
    v12 = (_QWORD *)(v10 + 288);
    *(_QWORD *)(v10 + 272) = 0;
    *(_QWORD *)(v10 + 280) = 1;
    do
    {
      if ( v12 )
      {
        *v12 = 0;
        v12[1] = 0;
        v12[2] = -4096;
      }
      v12 += 3;
    }
    while ( v12 != (_QWORD *)(v10 + 384) );
    *(_BYTE *)(v10 + 448) = 0;
  }
  v18 = v10;
  v22[0] = 2;
  v22[1] = 0;
  v23 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
  {
    sub_BD73F0((__int64)v22);
    v10 = v18;
  }
  v25[0] = v10;
  v24 = 0;
  v18 = 0;
  v21 = &unk_49DE8C0;
  v13 = sub_22BE090(a1, (__int64)&v21, &v19);
  v14 = v19;
  if ( !v13 )
  {
    v15 = *(_DWORD *)(a1 + 24);
    v16 = *(_DWORD *)(a1 + 16);
    v20 = v19;
    ++*(_QWORD *)a1;
    v17 = v16 + 1;
    if ( 4 * v17 >= 3 * v15 )
    {
      v15 *= 2;
    }
    else if ( v15 - *(_DWORD *)(a1 + 20) - v17 > v15 >> 3 )
    {
LABEL_28:
      *(_DWORD *)(a1 + 16) = v17;
      if ( *(_QWORD *)(v14 + 24) != -4096 )
        --*(_DWORD *)(a1 + 20);
      if ( *(_BYTE *)(v14 + 32) )
        *(_QWORD *)(v14 + 24) = 0;
      sub_22BDBB0((unsigned __int64 *)(v14 + 8), v22);
      *(_BYTE *)(v14 + 32) = v24;
      *(_QWORD *)(v14 + 40) = v25[0];
      v25[0] = 0;
      goto LABEL_21;
    }
    sub_22C1EB0(a1, v15);
    sub_22BE090(a1, (__int64)&v21, &v20);
    v14 = v20;
    v17 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_28;
  }
LABEL_21:
  sub_22C1BD0(v25);
  if ( !v24 )
  {
    v21 = &unk_49DB368;
    if ( v23 != -4096 && v23 != -8192 )
    {
      if ( v23 )
        sub_BD60C0(v22);
    }
  }
  sub_22C1BD0((unsigned __int64 *)&v18);
  return *(_QWORD *)(v14 + 40);
}
