// Function: sub_B23B20
// Address: 0xb23b20
//
void __fastcall sub_B23B20(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // ebx
  __int64 *v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // r14
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r11
  __int64 v13; // r15
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned int v16; // r8d
  int v17; // eax
  int v18; // esi
  __int64 *v21; // [rsp+20h] [rbp-80h]
  __int64 *v22; // [rsp+28h] [rbp-78h]
  __int64 *v23[4]; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v24[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v25; // [rsp+60h] [rbp-40h]

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v21 = a1 + 1;
    while ( 1 )
    {
      v5 = *a1;
      sub_B1C5B0(v23, a3, *v21);
      v6 = *((_DWORD *)v23[2] + 2);
      sub_B1C5B0(v24, a3, v5);
      if ( v6 >= *(_DWORD *)(v25 + 8) )
        break;
      v4 = *v21;
      if ( a1 != v21 )
        memmove(a1 + 1, a1, (char *)v21 - (char *)a1);
      *a1 = v4;
LABEL_7:
      if ( a2 == ++v21 )
        return;
    }
    v7 = v21;
    v8 = a3;
    v9 = *v21;
    while ( 1 )
    {
      v22 = v7;
      v13 = *(v7 - 1);
      sub_B1C5B0(v24, v8, v9);
      v14 = *((unsigned int *)v8 + 6);
      v15 = v8[1];
      v16 = *(_DWORD *)(v25 + 8);
      if ( (_DWORD)v14 )
      {
        v10 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v11 = (__int64 *)(v15 + 16LL * v10);
        v12 = *v11;
        if ( v13 == *v11 )
        {
LABEL_11:
          --v7;
          if ( v16 >= *((_DWORD *)v11 + 2) )
            goto LABEL_15;
          goto LABEL_12;
        }
        v17 = 1;
        while ( v12 != -4096 )
        {
          v18 = v17 + 1;
          v10 = (v14 - 1) & (v17 + v10);
          v11 = (__int64 *)(v15 + 16LL * v10);
          v12 = *v11;
          if ( v13 == *v11 )
            goto LABEL_11;
          v17 = v18;
        }
      }
      --v7;
      if ( v16 >= *(_DWORD *)(v15 + 16 * v14 + 8) )
      {
LABEL_15:
        a3 = v8;
        *v22 = v9;
        goto LABEL_7;
      }
LABEL_12:
      v7[1] = *v7;
    }
  }
}
