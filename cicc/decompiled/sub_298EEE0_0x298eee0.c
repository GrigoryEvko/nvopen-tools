// Function: sub_298EEE0
// Address: 0x298eee0
//
__int64 __fastcall sub_298EEE0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // r15
  __int64 result; // rax
  signed __int64 v4; // rbx
  unsigned __int64 v5; // rbx
  unsigned __int8 (__fastcall *v6)(_QWORD *); // r12
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r12d
  _QWORD *v11; // r14
  int v12; // eax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned int v15; // esi
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // r10d
  unsigned int i; // eax
  __int64 v23; // rdi
  unsigned int v24; // eax
  unsigned int v25; // ebx
  _QWORD *v26; // r12
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned int v29; // eax
  unsigned int v30; // [rsp+4h] [rbp-ECh]
  __int64 v31; // [rsp+10h] [rbp-E0h]
  __int64 v33; // [rsp+20h] [rbp-D0h]
  __int64 v34; // [rsp+28h] [rbp-C8h]
  _QWORD v35[11]; // [rsp+30h] [rbp-C0h] BYREF
  int v36; // [rsp+88h] [rbp-68h]

  v1 = *(_QWORD *)(a1 + 96);
  while ( 1 )
  {
    sub_2989F30((__int64)v35, v1 - 96);
    v2 = *(_QWORD *)(a1 + 96);
    v33 = *(_QWORD *)(v2 - 80);
    if ( ((v33 >> 1) & 3) != 0 )
      break;
    result = *(unsigned int *)(v2 - 64);
    v30 = result;
    if ( (_DWORD)result == v36 )
      return result;
    v25 = *(_DWORD *)(v2 - 64);
    v34 = *(_QWORD *)(v2 - 72);
    v26 = (_QWORD *)(v33 & 0xFFFFFFFFFFFFFFF8LL);
    v31 = *(_QWORD *)(v2 - 56);
    do
    {
      *(_DWORD *)(v2 - 64) = ++v25;
      v27 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
      v28 = *(_QWORD *)(v27 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v28 != v27 + 48 )
      {
        if ( !v28 )
LABEL_18:
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 <= 0xA )
        {
          if ( v25 == (unsigned int)sub_B46E30(v28 - 24) )
            break;
          continue;
        }
      }
      if ( !v25 )
        break;
    }
    while ( *(_QWORD *)(v26[1] + 32LL) == sub_B46EC0(v34, v25) );
    v4 = v33;
LABEL_6:
    while ( ((v4 >> 1) & 3) != 0 )
    {
      if ( ((v4 >> 1) & 3) == ((*(__int64 *)(v2 - 48) >> 1) & 3) )
        goto LABEL_22;
      v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = *(unsigned __int8 (__fastcall **)(_QWORD *))(v2 - 16);
      v7 = *(_QWORD *)(v5 + 32);
LABEL_9:
      v8 = sub_22DE030(*(_QWORD **)(v5 + 8), v7);
      v9 = *(_QWORD *)(v2 - 56);
      v35[0] = v8;
      v35[1] = v9;
      if ( v6(v35) )
        goto LABEL_22;
      v4 = *(_QWORD *)(v2 - 80);
      if ( (v4 & 6) != 0 )
        goto LABEL_5;
      v10 = *(_DWORD *)(v2 - 64);
      v11 = (_QWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
      do
      {
        *(_DWORD *)(v2 - 64) = ++v10;
        v13 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
        v14 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v14 == v13 + 48 )
          goto LABEL_19;
        if ( !v14 )
          goto LABEL_18;
        if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA )
LABEL_19:
          v12 = 0;
        else
          v12 = sub_B46E30(v14 - 24);
      }
      while ( v10 != v12 && *(_QWORD *)(v11[1] + 32LL) == sub_B46EC0(*(_QWORD *)(v2 - 72), v10) );
    }
    v15 = *(_DWORD *)(v2 - 64);
    if ( v15 != *(_DWORD *)(v2 - 32) )
    {
      v6 = *(unsigned __int8 (__fastcall **)(_QWORD *))(v2 - 16);
      v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      v7 = sub_B46EC0(*(_QWORD *)(v2 - 72), v15);
      goto LABEL_9;
    }
LABEL_22:
    v16 = v33 & 0xFFFFFFFFFFFFFFF8LL;
    if ( ((v33 >> 1) & 3) != 0 )
      v17 = *(_QWORD *)(v16 + 32);
    else
      v17 = sub_B46EC0(v34, v30);
    v18 = sub_22DE030(*(_QWORD **)(v16 + 8), v17);
    v19 = *(unsigned int *)(a1 + 32);
    v20 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v19 )
      goto LABEL_45;
    v21 = 1;
    for ( i = (v19 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)
                | ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)))); ; i = (v19 - 1) & v24 )
    {
      v23 = v20 + 24LL * i;
      if ( v18 == *(_QWORD **)v23 && v31 == *(_QWORD *)(v23 + 8) )
        break;
      if ( *(_QWORD *)v23 == -4096 && *(_QWORD *)(v23 + 8) == -4096 )
        goto LABEL_45;
      v24 = v21 + i;
      ++v21;
    }
    if ( v23 == v20 + 24 * v19 )
    {
LABEL_45:
      sub_298E970(a1, (__int64)v18, v31);
      v1 = *(_QWORD *)(a1 + 96);
    }
    else
    {
      v29 = *(_DWORD *)(v23 + 16);
      v1 = *(_QWORD *)(a1 + 96);
      if ( *(_DWORD *)(v1 - 8) > v29 )
      {
        *(_DWORD *)(v1 - 8) = v29;
        v1 = *(_QWORD *)(a1 + 96);
      }
    }
  }
  result = (v35[9] >> 1) & 3LL;
  if ( ((v33 >> 1) & 3) != (_DWORD)result )
  {
    v4 = *(_QWORD *)(v2 - 80);
    v34 = *(_QWORD *)(v2 - 72);
    v30 = *(_DWORD *)(v2 - 64);
    v31 = *(_QWORD *)(v2 - 56);
LABEL_5:
    v4 = v4 & 0xFFFFFFFFFFFFFFF9LL | 4;
    *(_QWORD *)(v2 - 80) = v4;
    goto LABEL_6;
  }
  return result;
}
