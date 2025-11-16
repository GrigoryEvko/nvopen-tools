// Function: sub_1BF2A10
// Address: 0x1bf2a10
//
__int64 __fastcall sub_1BF2A10(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  __int64 v6; // r12
  _QWORD *v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // rbx
  __int64 *v11; // r13
  __int64 v12; // rbx
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rax
  unsigned int v17; // edi
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rdx
  _QWORD *v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  char v24; // [rsp+1Fh] [rbp-41h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v5 = sub_13FD440(*(_QWORD *)a1);
  v6 = *(_QWORD *)(a2 + 48);
  v7 = (_QWORD *)(a1 + 488);
  v24 = v5;
  v26 = a2 + 40;
  v25 = a1 + 488;
  if ( v6 == a2 + 40 )
    return 1;
  while ( 1 )
  {
    if ( !v6 )
      BUG();
    v8 = 24LL * (*(_DWORD *)(v6 - 4) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v6 - 1) & 0x40) != 0 )
    {
      v9 = *(_QWORD *)(v6 - 32);
      v10 = (__int64 *)(v9 + v8);
    }
    else
    {
      v10 = (__int64 *)(v6 - 24);
      v9 = v6 - 24 - v8;
    }
    v11 = (__int64 *)v9;
    if ( (__int64 *)v9 != v10 )
      break;
LABEL_9:
    v12 = v6 - 24;
    if ( !(unsigned __int8)sub_15F2ED0(v6 - 24) )
      goto LABEL_26;
    if ( *(_BYTE *)(v6 - 8) != 54 )
      return 0;
    v13 = *(_QWORD **)(a3 + 16);
    v14 = *(_QWORD **)(a3 + 8);
    a2 = *(_QWORD *)(v6 - 48);
    if ( v13 == v14 )
    {
      v7 = &v14[*(unsigned int *)(a3 + 28)];
      if ( v14 == v7 )
      {
        v21 = *(_QWORD **)(a3 + 8);
      }
      else
      {
        do
        {
          if ( a2 == *v14 )
            break;
          ++v14;
        }
        while ( v7 != v14 );
        v21 = v7;
      }
    }
    else
    {
      v23 = *(_QWORD *)(v6 - 48);
      v22 = &v13[*(unsigned int *)(a3 + 24)];
      v14 = sub_16CC9F0(a3, a2);
      a2 = v23;
      v7 = v22;
      if ( v23 == *v14 )
      {
        v20 = *(_QWORD *)(a3 + 16);
        if ( v20 == *(_QWORD *)(a3 + 8) )
          a2 = *(unsigned int *)(a3 + 28);
        else
          a2 = *(unsigned int *)(a3 + 24);
        v21 = (_QWORD *)(v20 + 8 * a2);
      }
      else
      {
        v15 = *(_QWORD *)(a3 + 16);
        if ( v15 != *(_QWORD *)(a3 + 8) )
        {
          v14 = (_QWORD *)(v15 + 8LL * *(unsigned int *)(a3 + 24));
          goto LABEL_15;
        }
        v14 = (_QWORD *)(v15 + 8LL * *(unsigned int *)(a3 + 28));
        v21 = v14;
      }
    }
    while ( v21 != v14 && *v14 >= 0xFFFFFFFFFFFFFFFELL )
      ++v14;
LABEL_15:
    if ( v7 == v14 )
    {
      if ( !v24 )
      {
        v16 = *(_QWORD **)(a1 + 496);
        if ( *(_QWORD **)(a1 + 504) != v16 )
        {
LABEL_31:
          a2 = v6 - 24;
          sub_16CCBA0(v25, v6 - 24);
          goto LABEL_32;
        }
        a2 = (__int64)&v16[*(unsigned int *)(a1 + 516)];
        v17 = *(_DWORD *)(a1 + 516);
        if ( v16 != (_QWORD *)a2 )
        {
          v7 = 0;
          while ( v12 != *v16 )
          {
            if ( *v16 == -2 )
              v7 = v16;
            if ( (_QWORD *)a2 == ++v16 )
              goto LABEL_24;
          }
          goto LABEL_32;
        }
LABEL_56:
        if ( v17 < *(_DWORD *)(a1 + 512) )
        {
          *(_DWORD *)(a1 + 516) = v17 + 1;
          *(_QWORD *)a2 = v12;
          ++*(_QWORD *)(a1 + 488);
          goto LABEL_32;
        }
        goto LABEL_31;
      }
      goto LABEL_32;
    }
LABEL_26:
    if ( (unsigned __int8)sub_15F3040(v6 - 24) )
    {
      if ( *(_BYTE *)(v6 - 8) != 55 )
        return 0;
      v19 = *(_QWORD **)(a1 + 496);
      if ( *(_QWORD **)(a1 + 504) != v19 )
        goto LABEL_31;
      a2 = (__int64)&v19[*(unsigned int *)(a1 + 516)];
      v17 = *(_DWORD *)(a1 + 516);
      if ( v19 != (_QWORD *)a2 )
      {
        v7 = 0;
        while ( v12 != *v19 )
        {
          if ( *v19 == -2 )
            v7 = v19;
          if ( (_QWORD *)a2 == ++v19 )
          {
LABEL_24:
            if ( !v7 )
              goto LABEL_56;
            *v7 = v12;
            --*(_DWORD *)(a1 + 520);
            ++*(_QWORD *)(a1 + 488);
            goto LABEL_32;
          }
        }
        goto LABEL_32;
      }
      goto LABEL_56;
    }
    if ( sub_15F3330(v6 - 24) )
      return 0;
LABEL_32:
    v6 = *(_QWORD *)(v6 + 8);
    if ( v26 == v6 )
      return 1;
  }
  while ( *(_BYTE *)(*v11 + 16) > 0x10u || !(unsigned __int8)sub_1593DF0(*v11, a2, v9, v7) )
  {
    v11 += 3;
    if ( v10 == v11 )
      goto LABEL_9;
  }
  return 0;
}
