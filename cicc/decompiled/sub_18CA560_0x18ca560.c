// Function: sub_18CA560
// Address: 0x18ca560
//
__int64 __fastcall sub_18CA560(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 result; // rax
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // rbx
  char v13; // di
  unsigned int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rax
  __int64 v22; // [rsp+10h] [rbp-40h]
  unsigned int v23; // [rsp+1Ch] [rbp-34h]

  v2 = a1;
  result = sub_157F280(*(_QWORD *)(a1 + 40));
  v22 = v6;
  if ( v6 == result )
    return result;
  v7 = result;
  do
  {
    if ( v2 == v7 )
      goto LABEL_18;
    v23 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
    if ( !v23 )
    {
LABEL_27:
      v20 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v20 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v4, v5);
        v20 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v20) = v7;
      ++*(_DWORD *)(a2 + 8);
      if ( !v7 )
        BUG();
      goto LABEL_18;
    }
    v8 = v2;
    v9 = 0;
    do
    {
      if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
        v10 = *(_QWORD *)(v8 - 8);
      else
        v10 = v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
      v11 = *(_QWORD *)(24LL * *(unsigned int *)(v8 + 56) + v10 + 8 * v9 + 8);
      v12 = sub_1649C60(*(_QWORD *)(v10 + 24 * v9));
      v13 = *(_BYTE *)(v7 + 23) & 0x40;
      v14 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      if ( v14 )
      {
        v15 = 24LL * *(unsigned int *)(v7 + 56) + 8;
        v16 = 0;
        while ( 1 )
        {
          v17 = v7 - 24LL * v14;
          if ( v13 )
            v17 = *(_QWORD *)(v7 - 8);
          if ( v11 == *(_QWORD *)(v17 + v15) )
            break;
          ++v16;
          v15 += 8;
          if ( v14 == (_DWORD)v16 )
            goto LABEL_23;
        }
        v18 = 24 * v16;
        if ( v13 )
        {
LABEL_15:
          if ( v12 != sub_1649C60(*(_QWORD *)(*(_QWORD *)(v7 - 8) + v18)) )
            goto LABEL_25;
          goto LABEL_16;
        }
      }
      else
      {
LABEL_23:
        v18 = 0x17FFFFFFE8LL;
        if ( v13 )
          goto LABEL_15;
      }
      if ( v12 != sub_1649C60(*(_QWORD *)(v7 - 24LL * v14 + v18)) )
      {
LABEL_25:
        v2 = v8;
        goto LABEL_18;
      }
LABEL_16:
      v19 = ++v9;
    }
    while ( v23 > (unsigned int)v9 );
    v2 = v8;
    if ( v23 == v19 )
      goto LABEL_27;
LABEL_18:
    result = *(_QWORD *)(v7 + 32);
    if ( !result )
      BUG();
    v7 = 0;
    if ( *(_BYTE *)(result - 8) == 77 )
      v7 = result - 24;
  }
  while ( v22 != v7 );
  return result;
}
