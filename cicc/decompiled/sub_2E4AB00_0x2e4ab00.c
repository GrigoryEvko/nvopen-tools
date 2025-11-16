// Function: sub_2E4AB00
// Address: 0x2e4ab00
//
unsigned __int64 __fastcall sub_2E4AB00(
        __int64 a1,
        __int16 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int16 *a8,
        unsigned int a9)
{
  unsigned __int64 result; // rax
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdx
  unsigned int *v12; // rbx
  unsigned int v13; // r15d
  unsigned int *v14; // rax
  __int16 *v15; // rax
  __int16 *v16; // rdx
  int v17; // ecx
  unsigned int *v18; // r12
  bool v19; // r10
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // rdx
  char v24; // [rsp+Ch] [rbp-44h]
  _QWORD *v25; // [rsp+10h] [rbp-40h]

  result = (unsigned __int64)&a9;
  if ( a2 != a8 )
  {
    v10 = (_QWORD *)(a1 + 48);
    do
    {
      if ( !*(_QWORD *)(a1 + 88) )
      {
        v11 = *(unsigned int *)(a1 + 8);
        v12 = (unsigned int *)(*(_QWORD *)a1 + 4 * v11);
        if ( *(unsigned int **)a1 == v12 )
        {
          if ( v11 <= 7 )
          {
            v13 = a9;
LABEL_26:
            v23 = v11 + 1;
            if ( v23 > *(unsigned int *)(a1 + 12) )
            {
              sub_C8D5F0(a1, (const void *)(a1 + 16), v23, 4u, *(_QWORD *)a1, a6);
              v12 = (unsigned int *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8));
            }
            *v12 = v13;
            ++*(_DWORD *)(a1 + 8);
            goto LABEL_9;
          }
        }
        else
        {
          v13 = a9;
          v14 = *(unsigned int **)a1;
          while ( *v14 != a9 )
          {
            if ( v12 == ++v14 )
              goto LABEL_15;
          }
          if ( v12 != v14 )
            goto LABEL_9;
LABEL_15:
          if ( v11 <= 7 )
            goto LABEL_26;
          v18 = *(unsigned int **)a1;
          do
          {
            v21 = sub_B9AB10(v10, a1 + 56, v18);
            if ( v22 )
            {
              v19 = v21 || v22 == a1 + 56 || *v18 < *(_DWORD *)(v22 + 32);
              v24 = v19;
              v25 = (_QWORD *)v22;
              v20 = sub_22077B0(0x28u);
              *(_DWORD *)(v20 + 32) = *v18;
              sub_220F040(v24, v20, v25, (_QWORD *)(a1 + 56));
              ++*(_QWORD *)(a1 + 88);
            }
            ++v18;
          }
          while ( v12 != v18 );
        }
        *(_DWORD *)(a1 + 8) = 0;
      }
      sub_B99820((__int64)v10, &a9);
LABEL_9:
      v15 = a8;
      v16 = ++a8;
      v17 = *v15;
      result = (unsigned int)(a7 + v17);
      a7 += v17;
      if ( !(_WORD)v17 )
      {
        a8 = 0;
        v16 = 0;
      }
      a9 = result;
    }
    while ( a2 != v16 );
  }
  return result;
}
