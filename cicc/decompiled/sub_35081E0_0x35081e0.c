// Function: sub_35081E0
// Address: 0x35081e0
//
__int64 __fastcall sub_35081E0(_QWORD *a1, __int64 a2)
{
  unsigned int *v3; // rbx
  __int64 result; // rax
  __int64 v5; // rdx
  unsigned int *v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // rcx
  unsigned int v10; // esi
  __int64 v11; // r9
  __int16 *v12; // rbx
  int v13; // eax
  __int16 *v14; // rbx
  int v15; // r14d
  __int64 v16; // r9
  unsigned int v17; // esi
  unsigned __int16 *v18; // r13
  _QWORD *v19; // rax
  unsigned int *v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 v22; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int **)(a2 + 192);
  v20 = v3;
  result = sub_2E33140(a2);
  if ( v3 != (unsigned int *)result )
  {
    v6 = (unsigned int *)result;
    do
    {
      v7 = *a1;
      v8 = *((_QWORD *)v6 + 1);
      v9 = *((_QWORD *)v6 + 2);
      v10 = *v6;
      v11 = *(_QWORD *)(*a1 + 8LL) + 24LL * *v6;
      v12 = (__int16 *)(*(_QWORD *)(*a1 + 56LL) + 2LL * *(unsigned int *)(v11 + 4));
      v13 = *v12;
      if ( *v12 && (v14 = v12 + 1, (v8 & v9) != 0xFFFFFFFFFFFFFFFFLL) )
      {
        v15 = v13 + v10;
        v16 = *(unsigned int *)(v11 + 12);
        v17 = (unsigned __int16)(v13 + v10);
        v18 = (unsigned __int16 *)(*(_QWORD *)(v7 + 88) + 2 * v16);
        while ( 1 )
        {
          v19 = (_QWORD *)(*(_QWORD *)(v7 + 272) + 16LL * *v18);
          v5 = v9 & v19[1];
          if ( v5 | v8 & *v19 )
          {
            v21 = v8;
            v22 = v9;
            sub_3507B80(a1, v17, v5, v9, v8, v16);
            v8 = v21;
            v9 = v22;
          }
          result = (unsigned int)*v14;
          ++v18;
          ++v14;
          if ( !(_WORD)result )
            break;
          v15 += result;
          v7 = *a1;
          v17 = (unsigned __int16)v15;
        }
      }
      else
      {
        result = sub_3507B80(a1, v10, v5, v9, v8, v11);
      }
      v6 += 6;
    }
    while ( v20 != v6 );
  }
  return result;
}
