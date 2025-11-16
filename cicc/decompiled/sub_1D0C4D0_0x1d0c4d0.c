// Function: sub_1D0C4D0
// Address: 0x1d0c4d0
//
__int64 __fastcall sub_1D0C4D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v6; // r14
  __int64 result; // rax
  __int64 v9; // rsi
  unsigned int v11; // edx
  __int64 *v12; // rcx
  __int64 v13; // r9
  __int64 v14; // r8
  _QWORD *v15; // r15
  __int64 v16; // r12
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r11
  __int64 v20; // rdx
  _QWORD *v21; // rdx
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  int v24; // ecx
  int v25; // r10d
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned int v30; // [rsp+20h] [rbp-40h]
  unsigned __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  _QWORD *v33; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a3 + 48);
  v27 = *(_QWORD *)(a3 + 40);
  result = *(unsigned int *)(a2 + 720);
  if ( (_DWORD)result )
  {
    v9 = *(_QWORD *)(a2 + 704);
    v11 = (result - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v12 = (__int64 *)(v9 + 40LL * v11);
    v13 = *v12;
    if ( a1 == *v12 )
    {
LABEL_3:
      result = v9 + 40 * result;
      if ( v12 != (__int64 *)result )
      {
        v14 = v12[1];
        result = v14 + 8LL * *((unsigned int *)v12 + 4);
        v33 = (_QWORD *)result;
        if ( result != v14 )
        {
          v15 = (_QWORD *)v12[1];
          do
          {
            while ( 1 )
            {
              v16 = *v15;
              if ( !*(_BYTE *)(*v15 + 49LL) && (*(_DWORD *)(v16 + 40) == a6 || !a6) )
                break;
              if ( v33 == ++v15 )
                return result;
            }
            v30 = *(_DWORD *)(v16 + 40);
            result = sub_1FE7480(a3, *v15, a5);
            if ( result )
            {
              v19 = v30;
              v20 = *(unsigned int *)(a4 + 8);
              if ( (unsigned int)v20 >= *(_DWORD *)(a4 + 12) )
              {
                v26 = v30;
                v32 = result;
                sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v17, v18);
                v20 = *(unsigned int *)(a4 + 8);
                v19 = v26;
                result = v32;
              }
              v21 = (_QWORD *)(*(_QWORD *)a4 + 16 * v20);
              v31 = result;
              v21[1] = result;
              *v21 = v19;
              ++*(_DWORD *)(a4 + 8);
              sub_1DD5BA0(v27 + 16, result);
              v22 = *(_QWORD *)v31;
              v23 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v31 + 8) = v6;
              *(_QWORD *)v31 = v23 | v22 & 7;
              *(_QWORD *)(v23 + 8) = v31;
              result = *v6 & 7 | v31;
              *v6 = result;
            }
            *(_BYTE *)(v16 + 49) = 1;
            ++v15;
          }
          while ( v33 != v15 );
        }
      }
    }
    else
    {
      v24 = 1;
      while ( v13 != -8 )
      {
        v25 = v24 + 1;
        v11 = (result - 1) & (v24 + v11);
        v12 = (__int64 *)(v9 + 40LL * v11);
        v13 = *v12;
        if ( a1 == *v12 )
          goto LABEL_3;
        v24 = v25;
      }
    }
  }
  return result;
}
