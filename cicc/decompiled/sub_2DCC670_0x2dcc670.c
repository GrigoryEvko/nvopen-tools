// Function: sub_2DCC670
// Address: 0x2dcc670
//
void __fastcall sub_2DCC670(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // edx
  __int64 *v9; // rcx
  __int64 v10; // rdi
  int *v11; // r10
  int *v12; // r14
  __int64 v13; // r9
  unsigned __int64 v14; // rdx
  int *v15; // rbx
  int v16; // r11d
  int *v17; // rax
  int *v18; // rdx
  unsigned __int64 v19; // rdx
  _DWORD *v20; // rcx
  _DWORD *v21; // rdx
  _DWORD *v22; // rax
  int *v23; // r13
  bool v24; // r11
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // ecx
  int v29; // r10d
  _QWORD *v30; // [rsp+10h] [rbp-50h]
  char v31; // [rsp+1Ch] [rbp-44h]
  int v32; // [rsp+1Ch] [rbp-44h]
  int *v33; // [rsp+20h] [rbp-40h]
  _QWORD *v34; // [rsp+28h] [rbp-38h]

  if ( *(_DWORD *)(a1 + 32) )
  {
    v20 = *(_DWORD **)(a1 + 24);
    v21 = &v20[16 * (unsigned __int64)*(unsigned int *)(a1 + 40)];
    if ( v20 != v21 )
    {
      while ( 1 )
      {
        v22 = v20;
        if ( *v20 <= 0xFFFFFFFD )
          break;
        v20 += 16;
        if ( v21 == v20 )
          goto LABEL_2;
      }
      while ( v21 != v22 )
      {
        v22[14] = 0;
        v22 += 16;
        if ( v22 == v21 )
          break;
        while ( *v22 > 0xFFFFFFFD )
        {
          v22 += 16;
          if ( v21 == v22 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  *(_DWORD *)(a1 + 56) = 0;
  v4 = a1 + 104;
  sub_2DCAD80(*(_QWORD *)(a1 + 112));
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = a1 + 104;
  *(_QWORD *)(a1 + 128) = a1 + 104;
  *(_QWORD *)(a1 + 136) = 0;
  if ( a2 )
  {
    v6 = *(unsigned int *)(a1 + 168);
    v7 = *(_QWORD *)(a1 + 152);
    if ( (_DWORD)v6 )
    {
      v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v7 + 88LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
      {
LABEL_5:
        if ( v9 == (__int64 *)(v7 + 88 * v6) )
          return;
        v11 = (int *)v9[1];
        v33 = &v11[2 * *((unsigned int *)v9 + 4)];
        if ( v33 == v11 )
          return;
        v12 = v11 + 1;
        v34 = (_QWORD *)(a1 + 96);
LABEL_8:
        v13 = *(_QWORD *)(a1 + 48);
        v14 = *(unsigned int *)(a1 + 56);
        v15 = (int *)(v13 + 4 * v14);
        if ( (int *)v13 == v15 )
        {
          if ( v14 <= 7 )
          {
            v16 = *v12;
LABEL_19:
            v19 = v14 + 1;
            if ( v19 > *(unsigned int *)(a1 + 60) )
            {
              v32 = v16;
              sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v19, 4u, v5, v13);
              v16 = v32;
              v15 = (int *)(*(_QWORD *)(a1 + 48) + 4LL * *(unsigned int *)(a1 + 56));
            }
            *v15 = v16;
            ++*(_DWORD *)(a1 + 56);
LABEL_13:
            v18 = v12 + 2;
            if ( v33 != v12 + 1 )
              goto LABEL_14;
            return;
          }
        }
        else
        {
          v16 = *v12;
          v17 = *(int **)(a1 + 48);
          while ( *v17 != v16 )
          {
            if ( v15 == ++v17 )
              goto LABEL_18;
          }
          if ( v15 != v17 )
            goto LABEL_13;
LABEL_18:
          if ( v14 <= 7 )
            goto LABEL_19;
          v23 = *(int **)(a1 + 48);
          do
          {
            v26 = sub_BB8210(v34, a1 + 104, v23);
            if ( v27 )
            {
              v24 = v26 || v4 == v27 || *v23 < *(_DWORD *)(v27 + 32);
              v30 = (_QWORD *)v27;
              v31 = v24;
              v25 = sub_22077B0(0x28u);
              *(_DWORD *)(v25 + 32) = *v23;
              sub_220F040(v31, v25, v30, (_QWORD *)(a1 + 104));
              ++*(_QWORD *)(a1 + 136);
            }
            ++v23;
          }
          while ( v15 != v23 );
        }
        *(_DWORD *)(a1 + 56) = 0;
        sub_BB8160((__int64)v34, v12);
        while ( 1 )
        {
          v18 = v12 + 2;
          if ( v33 == v12 + 1 )
            return;
LABEL_14:
          v12 = v18;
          if ( !*(_QWORD *)(a1 + 136) )
            goto LABEL_8;
          sub_BB8160((__int64)v34, v18);
        }
      }
      v28 = 1;
      while ( v10 != -4096 )
      {
        v29 = v28 + 1;
        v8 = (v6 - 1) & (v28 + v8);
        v9 = (__int64 *)(v7 + 88LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_5;
        v28 = v29;
      }
    }
  }
}
