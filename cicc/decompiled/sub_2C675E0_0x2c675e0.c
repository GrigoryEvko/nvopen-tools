// Function: sub_2C675E0
// Address: 0x2c675e0
//
void __fastcall sub_2C675E0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // r14
  _QWORD *v5; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  char v10; // r15
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r9
  _QWORD *v14; // rdx
  __int64 v15; // r8
  _DWORD *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // r9
  bool v20; // cc
  _DWORD *v21; // rsi
  _DWORD *v22; // rcx
  _QWORD *v23; // [rsp+0h] [rbp-40h]
  _QWORD *v24; // [rsp+8h] [rbp-38h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = a3 << 6;
  v4 = a2 + v3;
  v5 = a1 + 1;
  *((_DWORD *)a1 + 2) = 0;
  a1[2] = 0;
  a1[3] = a1 + 1;
  a1[4] = a1 + 1;
  a1[5] = 0;
  if ( a2 + v3 != a2 )
  {
    v7 = a2;
    while ( 1 )
    {
      v8 = sub_2C67330(a1, (__int64)v5, v7);
      if ( v9 )
        break;
LABEL_4:
      v7 += 64;
      if ( v4 == v7 )
        return;
    }
    if ( !v8 && v5 != (_QWORD *)v9 )
    {
      v16 = *(_DWORD **)v7;
      v17 = 4LL * *(unsigned int *)(v7 + 8);
      v18 = 4LL * *(unsigned int *)(v9 + 40);
      v19 = *(_QWORD *)v7 + v17;
      v20 = v18 < v17;
      v21 = (_DWORD *)(*(_QWORD *)v7 + v18);
      v22 = *(_DWORD **)(v9 + 32);
      if ( !v20 )
        v21 = (_DWORD *)v19;
      if ( v16 == v21 )
      {
LABEL_20:
        v10 = v22 != (_DWORD *)(*(_QWORD *)(v9 + 32) + v18);
        goto LABEL_9;
      }
      while ( *v16 >= *v22 )
      {
        if ( *v16 > *v22 )
        {
          v10 = 0;
          goto LABEL_9;
        }
        ++v16;
        ++v22;
        if ( v21 == v16 )
          goto LABEL_20;
      }
    }
    v10 = 1;
LABEL_9:
    v24 = (_QWORD *)v9;
    v11 = sub_22077B0(0x60u);
    v14 = v24;
    v15 = v11;
    *(_QWORD *)(v11 + 32) = v11 + 48;
    *(_QWORD *)(v11 + 40) = 0xC00000000LL;
    if ( *(_DWORD *)(v7 + 8) )
    {
      v23 = v24;
      v25 = v11;
      sub_2C4D390(v11 + 32, v7, (__int64)v14, v12, v11, v13);
      v14 = v23;
      v15 = v25;
    }
    sub_220F040(v10, v15, v14, v5);
    ++a1[5];
    goto LABEL_4;
  }
}
