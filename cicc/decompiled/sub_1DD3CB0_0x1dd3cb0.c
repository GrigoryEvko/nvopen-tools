// Function: sub_1DD3CB0
// Address: 0x1dd3cb0
//
__int64 __fastcall sub_1DD3CB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __int64 *a6, unsigned int *a7)
{
  int *v8; // rbx
  __int64 result; // rax
  __int64 v10; // r12
  __int64 v11; // r15
  int v12; // r9d
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  _DWORD *v15; // rcx
  unsigned int v16; // r8d
  __int64 v17; // rdx
  __int64 v18; // r13
  _BOOL4 v19; // r8d
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned int v23; // eax
  int *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r15
  _BOOL4 v28; // r14d
  __int64 v29; // rdx
  __int64 v30; // r13
  _BOOL4 v31; // r8d
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-78h]
  _BOOL4 v34; // [rsp+10h] [rbp-70h]
  int *v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  _BOOL4 v37; // [rsp+18h] [rbp-68h]
  int *v41; // [rsp+38h] [rbp-48h]
  int v42[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v8 = *(int **)(a2 + 48);
  result = (__int64)&v8[*(unsigned int *)(a2 + 56)];
  v41 = (int *)result;
  if ( (int *)result != v8 )
  {
    v10 = a3 + 80;
    v11 = a3;
    do
    {
      v42[0] = *v8;
      sub_1DD2790(a1, a4, v42[0], a6, a5, a7);
      if ( *(_QWORD *)(v11 + 120) )
      {
        result = sub_BB80C0(v10, v42);
        v18 = v17;
        if ( v17 )
        {
          v19 = 1;
          if ( !result && v17 != v11 + 88 )
            v19 = v42[0] < *(_DWORD *)(v17 + 32);
          v34 = v19;
          v20 = sub_22077B0(40);
          *(_DWORD *)(v20 + 32) = v42[0];
          result = sub_220F040(v34, v20, v18, v11 + 88);
          ++*(_QWORD *)(v11 + 120);
        }
      }
      else
      {
        v13 = *(unsigned int *)(v11 + 8);
        v14 = *(_QWORD *)v11;
        v15 = (_DWORD *)(*(_QWORD *)v11 + 4 * v13);
        v16 = *(_DWORD *)(v11 + 8);
        if ( *(_DWORD **)v11 == v15 )
          goto LABEL_14;
        result = *(_QWORD *)v11;
        while ( *(_DWORD *)result != v42[0] )
        {
          result += 4;
          if ( v15 == (_DWORD *)result )
            goto LABEL_14;
        }
        if ( v15 == (_DWORD *)result )
        {
LABEL_14:
          v36 = v11 + 88;
          if ( v13 <= 0xF )
          {
            if ( v16 >= *(_DWORD *)(v11 + 12) )
            {
              sub_16CD150(v11, (const void *)(v11 + 16), 0, 4, v16, v12);
              v15 = (_DWORD *)(*(_QWORD *)v11 + 4LL * *(unsigned int *)(v11 + 8));
            }
            result = (unsigned int)v42[0];
            *v15 = v42[0];
            ++*(_DWORD *)(v11 + 8);
          }
          else
          {
            v35 = v8;
            v21 = v11;
            v33 = a1;
            while ( 1 )
            {
              v24 = (int *)(v14 + 4 * v13 - 4);
              v25 = sub_BB80C0(v10, v24);
              v27 = v26;
              if ( v26 )
              {
                v28 = 1;
                if ( !v25 && v26 != v36 )
                  v28 = *v24 < *(_DWORD *)(v26 + 32);
                v22 = sub_22077B0(40);
                *(_DWORD *)(v22 + 32) = *v24;
                sub_220F040(v28, v22, v27, v36);
                ++*(_QWORD *)(v21 + 120);
              }
              v23 = *(_DWORD *)(v21 + 8) - 1;
              *(_DWORD *)(v21 + 8) = v23;
              if ( !v23 )
                break;
              v14 = *(_QWORD *)v21;
              v13 = v23;
            }
            v11 = v21;
            a1 = v33;
            result = sub_BB80C0(v10, v42);
            v8 = v35;
            v30 = v29;
            if ( v29 )
            {
              v31 = 1;
              if ( !result && v29 != v11 + 88 )
                v31 = v42[0] < *(_DWORD *)(v29 + 32);
              v37 = v31;
              v32 = sub_22077B0(40);
              *(_DWORD *)(v32 + 32) = v42[0];
              result = sub_220F040(v37, v32, v30, v11 + 88);
              ++*(_QWORD *)(v11 + 120);
            }
          }
        }
      }
      ++v8;
    }
    while ( v8 != v41 );
  }
  return result;
}
