// Function: sub_3737680
// Address: 0x3737680
//
unsigned __int8 *__fastcall sub_3737680(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // r13
  unsigned __int8 *v5; // rax
  int v6; // edx
  __int64 v7; // rdi
  __int64 v8; // rsi
  unsigned int v9; // edx
  unsigned __int8 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  unsigned int v14; // ecx
  unsigned __int8 **v15; // rdx
  unsigned __int8 *v16; // r8
  int v18; // r8d
  int v19; // eax
  __int64 v20; // rsi
  int v21; // ecx
  unsigned int v22; // edx
  unsigned __int8 **v23; // rax
  unsigned __int8 *v24; // rdi
  int v25; // edx
  int v26; // r9d
  int v27; // eax
  int v28; // r8d

  if ( !sub_3734FE0(a1) || (v4 = a1 + 672, (unsigned __int8)sub_321F6A0(*(_QWORD *)(a1 + 208), a2)) )
    v4 = *(_QWORD *)(a1 + 216) + 400LL;
  v5 = sub_AF34D0(a2);
  v6 = *(_DWORD *)(v4 + 24);
  v7 = *(_QWORD *)(v4 + 8);
  if ( v6 )
  {
    v8 = (unsigned int)(v6 - 1);
    v9 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = *(unsigned __int8 **)(v7 + 16LL * v9);
    if ( v5 == v10 )
    {
LABEL_5:
      if ( !sub_3734FE0(a1) || (unsigned __int8)sub_321F6A0(*(_QWORD *)(a1 + 208), v8) )
        v11 = *(_QWORD *)(a1 + 216) + 400LL;
      else
        v11 = a1 + 672;
      v12 = *(_QWORD *)(v11 + 8);
      v13 = *(unsigned int *)(v11 + 24);
      if ( (_DWORD)v13 )
      {
        v14 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = (unsigned __int8 **)(v12 + 16LL * v14);
        v16 = *v15;
        if ( a2 == *v15 )
        {
LABEL_9:
          if ( v15 != (unsigned __int8 **)(v12 + 16 * v13) )
            return v15[1];
        }
        else
        {
          v25 = 1;
          while ( v16 != (unsigned __int8 *)-4096LL )
          {
            v26 = v25 + 1;
            v14 = (v13 - 1) & (v25 + v14);
            v15 = (unsigned __int8 **)(v12 + 16LL * v14);
            v16 = *v15;
            if ( a2 == *v15 )
              goto LABEL_9;
            v25 = v26;
          }
        }
      }
    }
    else
    {
      v18 = 1;
      while ( v10 != (unsigned __int8 *)-4096LL )
      {
        v9 = v8 & (v18 + v9);
        v10 = *(unsigned __int8 **)(v7 + 16LL * v9);
        if ( v5 == v10 )
          goto LABEL_5;
        ++v18;
      }
    }
  }
  v19 = *(_DWORD *)(a1 + 664);
  v20 = *(_QWORD *)(a1 + 648);
  if ( v19 )
  {
    v21 = v19 - 1;
    v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = (unsigned __int8 **)(v20 + 16LL * v22);
    v24 = *v23;
    if ( a2 == *v23 )
      return v23[1];
    v27 = 1;
    while ( v24 != (unsigned __int8 *)-4096LL )
    {
      v28 = v27 + 1;
      v22 = v21 & (v27 + v22);
      v23 = (unsigned __int8 **)(v20 + 16LL * v22);
      v24 = *v23;
      if ( a2 == *v23 )
        return v23[1];
      v27 = v28;
    }
  }
  return 0;
}
