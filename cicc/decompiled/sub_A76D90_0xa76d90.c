// Function: sub_A76D90
// Address: 0xa76d90
//
__int64 __fastcall sub_A76D90(_QWORD *a1, __int64 a2)
{
  void *v3; // rdx
  int v4; // ebx
  int v5; // eax
  int v6; // r14d
  _DWORD *v7; // rdx
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  _DWORD *v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  _WORD *v17; // rdx
  __int64 v19; // rax
  __int64 v20[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF

  v3 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0xEu )
  {
    sub_CB6200(a2, "AttributeList[\n", 15);
  }
  else
  {
    qmemcpy(v3, "AttributeList[\n", 15);
    *(_QWORD *)(a2 + 32) += 15LL;
  }
  v4 = -1;
  v5 = sub_A74480((__int64)a1);
  v6 = v5 - 1;
  if ( v5 )
  {
    while ( !sub_A74490(a1, v4) )
    {
LABEL_20:
      if ( v6 == ++v4 )
        goto LABEL_21;
    }
    v7 = *(_DWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 3u )
    {
      sub_CB6200(a2, "  { ", 4);
      v8 = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *v7 = 544940064;
      v8 = *(_QWORD *)(a2 + 32) + 4LL;
      *(_QWORD *)(a2 + 32) = v8;
    }
    v9 = *(_QWORD *)(a2 + 24) - v8;
    if ( v4 )
    {
      if ( v4 == -1 )
      {
        if ( v9 > 7 )
        {
          *(_QWORD *)v8 = 0x6E6F6974636E7566LL;
          v13 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
          v19 = *(_QWORD *)(a2 + 24);
          *(_QWORD *)(a2 + 32) = v13;
          if ( (unsigned __int64)(v19 - (_QWORD)v13) > 3 )
            goto LABEL_15;
          goto LABEL_25;
        }
        sub_CB6200(a2, "function", 8);
        v13 = *(_DWORD **)(a2 + 32);
      }
      else
      {
        if ( v9 <= 3 )
        {
          v10 = sub_CB6200(a2, "arg(", 4);
        }
        else
        {
          *(_DWORD *)v8 = 677868129;
          v10 = a2;
          *(_QWORD *)(a2 + 32) += 4LL;
        }
        v11 = sub_CB59D0(v10, (unsigned int)(v4 - 1));
        v12 = *(_BYTE **)(v11 + 32);
        if ( *(_BYTE **)(v11 + 24) == v12 )
        {
          sub_CB6200(v11, ")", 1);
        }
        else
        {
          *v12 = 41;
          ++*(_QWORD *)(v11 + 32);
        }
        v13 = *(_DWORD **)(a2 + 32);
      }
    }
    else if ( v9 <= 5 )
    {
      sub_CB6200(a2, "return", 6);
      v13 = *(_DWORD **)(a2 + 32);
    }
    else
    {
      *(_DWORD *)v8 = 1970562418;
      *(_WORD *)(v8 + 4) = 28274;
      v13 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 6LL);
      *(_QWORD *)(a2 + 32) = v13;
    }
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v13 > 3u )
    {
LABEL_15:
      *v13 = 540949792;
      v14 = a2;
      *(_QWORD *)(a2 + 32) += 4LL;
      goto LABEL_16;
    }
LABEL_25:
    v14 = sub_CB6200(a2, " => ", 4);
LABEL_16:
    sub_A76D50(v20, a1, v4, 0);
    v15 = sub_CB6200(v14, v20[0], v20[1]);
    v16 = *(_QWORD *)(v15 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 2 )
    {
      sub_CB6200(v15, " }\n", 3);
    }
    else
    {
      *(_BYTE *)(v16 + 2) = 10;
      *(_WORD *)v16 = 32032;
      *(_QWORD *)(v15 + 32) += 3LL;
    }
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0], v21 + 1);
    goto LABEL_20;
  }
LABEL_21:
  v17 = *(_WORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v17 <= 1u )
    return sub_CB6200(a2, "]\n", 2);
  *v17 = 2653;
  *(_QWORD *)(a2 + 32) += 2LL;
  return 2653;
}
