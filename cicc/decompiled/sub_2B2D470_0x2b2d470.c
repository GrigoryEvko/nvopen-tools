// Function: sub_2B2D470
// Address: 0x2b2d470
//
__int64 __fastcall sub_2B2D470(__int64 **a1, unsigned int *a2, int a3)
{
  __int64 *v6; // rcx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned int v11; // edx
  __int64 v12; // r15
  unsigned int *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // ebx
  __int64 v19; // rdx
  unsigned int v20; // r15d
  __int64 *v21; // rbx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r13
  __int64 v25; // rdi
  int v26; // eax
  __int64 v27; // rax
  int v29; // r9d
  _QWORD *v30; // r13
  char v31; // [rsp+Fh] [rbp-41h]
  __int64 v32; // [rsp+10h] [rbp-40h] BYREF
  __int64 v33; // [rsp+18h] [rbp-38h]

  if ( a2[26] != 3
    || (v30 = (_QWORD *)(*(_QWORD *)a2 + 8LL * a2[2]),
        v30 != sub_2B0BF30(*(_QWORD **)a2, (__int64)v30, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0)) )
  {
    v6 = *a1;
    v7 = *(_QWORD *)(**(_QWORD **)a2 + 8LL);
    v8 = (*a1)[23];
    v9 = *(_QWORD *)(v8 + 3528);
    v10 = *(unsigned int *)(v8 + 3544);
    if ( (_DWORD)v10 )
    {
      v11 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = v9 + 24LL * v11;
      v13 = *(unsigned int **)v12;
      if ( a2 == *(unsigned int **)v12 )
      {
LABEL_4:
        if ( v12 != v9 + 24 * v10 )
        {
          v14 = sub_BCCE00(*(_QWORD **)v7, *(_DWORD *)(v12 + 8));
          v6 = *a1;
          v7 = v14;
          v15 = **a1;
          v31 = *(_BYTE *)(v12 + 16);
          if ( v15 != v14 )
            goto LABEL_6;
          return 0;
        }
      }
      else
      {
        v29 = 1;
        while ( v13 != (unsigned int *)-4096LL )
        {
          v11 = (v10 - 1) & (v29 + v11);
          v12 = v9 + 24LL * v11;
          v13 = *(unsigned int **)v12;
          if ( a2 == *(unsigned int **)v12 )
            goto LABEL_4;
          ++v29;
        }
      }
    }
    v15 = *v6;
    v31 = 1;
    if ( *v6 != v7 )
    {
LABEL_6:
      v16 = sub_9208B0(*(_QWORD *)(v6[23] + 3344), v15);
      v33 = v17;
      v32 = v16;
      v18 = sub_CA1930(&v32);
      v32 = sub_9208B0(*(_QWORD *)((*a1)[23] + 3344), v7);
      v33 = v19;
      v20 = 39 - ((v31 == 0) - 1);
      if ( v18 <= (unsigned int)sub_CA1930(&v32) )
        v20 = 38;
      v21 = (__int64 *)(*a1)[14];
      v22 = *(unsigned __int8 *)(v7 + 8);
      if ( (_BYTE)v22 == 17 )
      {
        v23 = a3 * *(_DWORD *)(v7 + 32);
      }
      else
      {
        v23 = a3;
        if ( (unsigned int)(v22 - 17) > 1 )
        {
LABEL_11:
          v24 = sub_BCDA70((__int64 *)v7, v23);
          v25 = **a1;
          v26 = *(unsigned __int8 *)(v25 + 8);
          if ( (_BYTE)v26 == 17 )
          {
            a3 *= *(_DWORD *)(v25 + 32);
          }
          else if ( (unsigned int)(v26 - 17) > 1 )
          {
            goto LABEL_14;
          }
          v25 = **(_QWORD **)(v25 + 16);
LABEL_14:
          v27 = sub_BCDA70((__int64 *)v25, a3);
          return sub_DFD060(v21, v20, v27, v24);
        }
      }
      v7 = **(_QWORD **)(v7 + 16);
      goto LABEL_11;
    }
    return 0;
  }
  return 0;
}
