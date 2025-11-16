// Function: sub_2E2D830
// Address: 0x2e2d830
//
unsigned __int64 __fastcall sub_2E2D830(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 *a6,
        unsigned __int8 *a7)
{
  unsigned int *v7; // rbx
  unsigned __int64 result; // rax
  __int64 v10; // r8
  unsigned __int64 v11; // rdx
  unsigned int *v12; // r15
  unsigned int v13; // r13d
  int *v14; // rbx
  char v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  unsigned int *v20; // [rsp+0h] [rbp-90h]
  _QWORD *v21; // [rsp+10h] [rbp-80h]
  _QWORD *v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+20h] [rbp-70h]
  unsigned int *v27; // [rsp+48h] [rbp-48h]
  unsigned int v28[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v7 = *(unsigned int **)(a2 + 32);
  result = (unsigned __int64)&v7[*(unsigned int *)(a2 + 40)];
  v21 = (_QWORD *)(a3 + 80);
  v27 = (unsigned int *)result;
  while ( v27 != v7 )
  {
    v28[0] = *v7;
    sub_2E2CB70(a1, a4, v28[0], a6, a5, a7);
    if ( !*(_QWORD *)(a3 + 120) )
    {
      v11 = *(unsigned int *)(a3 + 8);
      v12 = (unsigned int *)(*(_QWORD *)a3 + 4 * v11);
      if ( *(unsigned int **)a3 == v12 )
      {
        if ( v11 <= 0xF )
        {
          v13 = v28[0];
LABEL_24:
          result = *(unsigned int *)(a3 + 12);
          v19 = v11 + 1;
          if ( v19 > result )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v19, 4u, v10, *(_QWORD *)a3);
            result = *(_QWORD *)a3;
            v12 = (unsigned int *)(*(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8));
          }
          *v12 = v13;
          ++*(_DWORD *)(a3 + 8);
          goto LABEL_8;
        }
      }
      else
      {
        v13 = v28[0];
        result = *(_QWORD *)a3;
        while ( *(_DWORD *)result != v28[0] )
        {
          result += 4LL;
          if ( v12 == (unsigned int *)result )
            goto LABEL_13;
        }
        if ( v12 != (unsigned int *)result )
          goto LABEL_8;
LABEL_13:
        if ( v11 <= 0xF )
          goto LABEL_24;
        v20 = v7;
        v14 = *(int **)a3;
        v23 = *(_QWORD *)a3 + 4 * v11;
        do
        {
          v17 = sub_BB8210(v21, a3 + 88, v14);
          if ( v18 )
          {
            v15 = v17 || v18 == a3 + 88 || *v14 < *(_DWORD *)(v18 + 32);
            v22 = (_QWORD *)v18;
            v16 = sub_22077B0(0x28u);
            *(_DWORD *)(v16 + 32) = *v14;
            sub_220F040(v15, v16, v22, (_QWORD *)(a3 + 88));
            ++*(_QWORD *)(a3 + 120);
          }
          ++v14;
        }
        while ( (int *)v23 != v14 );
        v7 = v20;
      }
      *(_DWORD *)(a3 + 8) = 0;
    }
    result = sub_BB8160((__int64)v21, (int *)v28);
LABEL_8:
    ++v7;
  }
  return result;
}
