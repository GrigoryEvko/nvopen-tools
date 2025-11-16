// Function: sub_2F39760
// Address: 0x2f39760
//
unsigned __int64 *__fastcall sub_2F39760(__int64 a1)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 *v3; // r12
  __int64 v4; // rdi
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // r14
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  unsigned __int64 *result; // rax
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rsi
  unsigned __int64 *v13; // r14
  unsigned __int64 *v14; // r15
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  unsigned __int64 *v18; // rdx
  __int64 *v19; // r12
  __int64 *v20; // r8
  __int64 v21; // rax
  unsigned __int64 *v22; // r13
  unsigned __int64 *v23; // r14
  __int64 v24; // rax
  unsigned __int64 *v25; // r15
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rax
  unsigned __int64 *v28; // [rsp+8h] [rbp-38h]
  __int64 *v29; // [rsp+8h] [rbp-38h]

  v2 = *(unsigned __int64 **)(a1 + 920);
  v3 = *(unsigned __int64 **)(a1 + 3368);
  *(_QWORD *)(a1 + 912) = v2;
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 904);
    if ( v3 != v2 )
    {
      v5 = v3;
      if ( (*(_BYTE *)v3 & 4) == 0 && (*((_BYTE *)v3 + 44) & 8) != 0 )
      {
        do
          v5 = (unsigned __int64 *)v5[1];
        while ( (*((_BYTE *)v5 + 44) & 8) != 0 );
      }
      v6 = (unsigned __int64 *)v5[1];
      if ( v3 != v6 && v2 != v6 )
      {
        sub_2E310C0((__int64 *)(v4 + 40), (__int64 *)(v4 + 40), (__int64)v3, v5[1]);
        if ( v6 != v3 )
        {
          v7 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v3 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v6;
          *v6 = *v6 & 7 | *v3 & 0xFFFFFFFFFFFFFFF8LL;
          v8 = *v2;
          *(_QWORD *)(v7 + 8) = v2;
          v8 &= 0xFFFFFFFFFFFFFFF8LL;
          *v3 = v8 | *v3 & 7;
          *(_QWORD *)(v8 + 8) = v3;
          *v2 = v7 | *v2 & 7;
        }
      }
    }
  }
  result = *(unsigned __int64 **)(a1 + 3584);
  v10 = (__int64)(*(_QWORD *)(a1 + 3592) - (_QWORD)result) >> 3;
  if ( (_DWORD)v10 )
  {
    v11 = 0;
    v12 = *(_QWORD *)(a1 + 904);
    result = (unsigned __int64 *)*result;
    if ( result )
      goto LABEL_12;
LABEL_25:
    result = (unsigned __int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 16) + 888LL))(
                                   *(_QWORD *)(a1 + 16),
                                   v12,
                                   *(_QWORD *)(a1 + 920));
    if ( v11 )
    {
      while ( ++v11 != (unsigned int)v10 )
      {
LABEL_24:
        v12 = *(_QWORD *)(a1 + 904);
        result = *(unsigned __int64 **)(*(_QWORD *)(a1 + 3584) + 8 * v11);
        if ( !result )
          goto LABEL_25;
LABEL_12:
        v13 = (unsigned __int64 *)*result;
        v14 = *(unsigned __int64 **)(a1 + 920);
        if ( (unsigned __int64 *)*result != v14 )
        {
          if ( !v13 )
            BUG();
          v15 = (unsigned __int64 *)*result;
          if ( (*(_BYTE *)v13 & 4) == 0 && (*((_BYTE *)v13 + 44) & 8) != 0 )
          {
            do
              v15 = (unsigned __int64 *)v15[1];
            while ( (*((_BYTE *)v15 + 44) & 8) != 0 );
          }
          result = (unsigned __int64 *)v15[1];
          if ( v13 != result && v14 != result )
          {
            v28 = result;
            sub_2E310C0((__int64 *)(v12 + 40), (__int64 *)(v12 + 40), (__int64)v13, (__int64)result);
            result = v28;
            if ( v28 != v14 && v28 != v13 )
            {
              v16 = *v28 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v28;
              *v28 = *v28 & 7 | *v13 & 0xFFFFFFFFFFFFFFF8LL;
              v17 = *v14;
              *(_QWORD *)(v16 + 8) = v14;
              v17 &= 0xFFFFFFFFFFFFFFF8LL;
              *v13 = v17 | *v13 & 7;
              *(_QWORD *)(v17 + 8) = v13;
              result = (unsigned __int64 *)(v16 | *v14 & 7);
              *v14 = (unsigned __int64)result;
            }
          }
        }
        if ( !v11 )
          goto LABEL_26;
      }
    }
    else
    {
LABEL_26:
      result = (unsigned __int64 *)(**(_QWORD **)(a1 + 920) & 0xFFFFFFFFFFFFFFF8LL);
      if ( !result )
        BUG();
      v18 = (unsigned __int64 *)*result;
      if ( (*result & 4) == 0 && (*(_BYTE *)((**(_QWORD **)(a1 + 920) & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          result = (unsigned __int64 *)((unsigned __int64)v18 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (*(_BYTE *)(((unsigned __int64)v18 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
            break;
          v18 = (unsigned __int64 *)*result;
        }
      }
      ++v11;
      *(_QWORD *)(a1 + 912) = result;
      if ( v11 != (unsigned int)v10 )
        goto LABEL_24;
    }
  }
  v19 = *(__int64 **)(a1 + 3352);
  v20 = *(__int64 **)(a1 + 3344);
  if ( v20 != v19 )
  {
    do
    {
      v21 = *(v19 - 1);
      v19 -= 2;
      v22 = (unsigned __int64 *)*v19;
      if ( !v21 )
        BUG();
      if ( (*(_BYTE *)v21 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v21 + 44) & 8) != 0 )
          v21 = *(_QWORD *)(v21 + 8);
      }
      v23 = *(unsigned __int64 **)(v21 + 8);
      if ( v22 != v23 )
      {
        if ( !v22 )
          BUG();
        v24 = *v19;
        if ( (*(_BYTE *)v22 & 4) == 0 && (*((_BYTE *)v22 + 44) & 8) != 0 )
        {
          do
            v24 = *(_QWORD *)(v24 + 8);
          while ( (*(_BYTE *)(v24 + 44) & 8) != 0 );
        }
        v25 = *(unsigned __int64 **)(v24 + 8);
        if ( v22 != v25 && v23 != v25 )
        {
          v29 = v20;
          sub_2E310C0(
            (__int64 *)(*(_QWORD *)(a1 + 904) + 40LL),
            (__int64 *)(*(_QWORD *)(a1 + 904) + 40LL),
            *v19,
            *(_QWORD *)(v24 + 8));
          v20 = v29;
          if ( v25 != v22 )
          {
            v26 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v22 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v25;
            *v25 = *v25 & 7 | *v22 & 0xFFFFFFFFFFFFFFF8LL;
            v27 = *v23;
            *(_QWORD *)(v26 + 8) = v23;
            v27 &= 0xFFFFFFFFFFFFFFF8LL;
            *v22 = v27 | *v22 & 7;
            *(_QWORD *)(v27 + 8) = v22;
            *v23 = v26 | *v23 & 7;
          }
        }
      }
    }
    while ( v20 != v19 );
    result = *(unsigned __int64 **)(a1 + 3344);
    if ( *(unsigned __int64 **)(a1 + 3352) != result )
      *(_QWORD *)(a1 + 3352) = result;
  }
  *(_QWORD *)(a1 + 3368) = 0;
  return result;
}
