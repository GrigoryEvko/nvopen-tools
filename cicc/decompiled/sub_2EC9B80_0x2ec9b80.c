// Function: sub_2EC9B80
// Address: 0x2ec9b80
//
__int64 __fastcall sub_2EC9B80(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 *v6; // rdi
  unsigned int v9; // r15d
  int v10; // eax
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // eax
  __int64 v18; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // edx
  unsigned int v23; // eax
  unsigned int v24; // [rsp-8h] [rbp-48h]
  int v25; // [rsp+Ch] [rbp-34h]

  v6 = *(__int64 **)(a2 + 16);
  if ( !v6 )
  {
    *(_BYTE *)(a3 + 24) = 16;
    return 1;
  }
  v9 = sub_2EC9A80(v6, *(_BYTE *)(a2 + 25));
  v10 = sub_2EC9A80(*(__int64 **)(a3 + 16), *(_BYTE *)(a3 + 25));
  if ( (unsigned __int8)sub_2EC9250(v10, v9, a3, a2, 2u) )
    goto LABEL_18;
  v11 = *(_QWORD *)(a1 + 136);
  if ( *(_BYTE *)(v11 + 4016) )
  {
    v9 = v24;
    if ( (unsigned __int8)sub_2EC98E0(
                            (_WORD *)(a3 + 26),
                            (__int16 *)(a2 + 26),
                            a3,
                            a2,
                            3u,
                            *(_QWORD *)(a1 + 24),
                            *(_QWORD *)(v11 + 32)) )
      goto LABEL_18;
    v20 = *(_QWORD *)(a1 + 136);
    if ( *(_BYTE *)(v20 + 4016) )
    {
      if ( (unsigned __int8)sub_2EC98E0(
                              (_WORD *)(a3 + 30),
                              (__int16 *)(a2 + 30),
                              a3,
                              a2,
                              4u,
                              *(_QWORD *)(a1 + 24),
                              *(_QWORD *)(v20 + 32)) )
        goto LABEL_18;
    }
  }
  if ( a4 )
  {
    if ( *(_BYTE *)(a1 + 52) && !a4[42] && (unsigned __int8)sub_2EC9280(a3, a2, a4) )
      goto LABEL_18;
    v9 = sub_2EC8BB0((__int64)a4, *(_QWORD *)(a2 + 16));
    v12 = sub_2EC8BB0((__int64)a4, *(_QWORD *)(a3 + 16));
    if ( (unsigned __int8)sub_2EC9220(v12, v9, a3, a2, 5u) )
      goto LABEL_18;
  }
  v13 = *(_QWORD *)(a1 + 136);
  v14 = *(_QWORD *)(v13 + 3520);
  v15 = *(_QWORD *)(v13 + 3528);
  v16 = v14;
  if ( *(_BYTE *)(a2 + 25) )
    v16 = v15;
  if ( !*(_BYTE *)(a3 + 25) )
    v15 = v14;
  v9 = sub_2EC9250(*(_QWORD *)(a3 + 16) == v15, *(_QWORD *)(a2 + 16) == v16, a3, a2, 6u);
  if ( (_BYTE)v9 )
    goto LABEL_18;
  if ( a4 )
  {
    v25 = sub_2EC9A60(*(_QWORD *)(a2 + 16), *(_BYTE *)(a2 + 25));
    v17 = sub_2EC9A60(*(_QWORD *)(a3 + 16), *(_BYTE *)(a3 + 25));
    if ( !(unsigned __int8)sub_2EC9220(v17, v25, a3, a2, 7u) )
    {
      v18 = *(_QWORD *)(a1 + 136);
      if ( !*(_BYTE *)(v18 + 4016) )
      {
LABEL_17:
        sub_2EC8FB0(a3, v18, *(_QWORD *)(a1 + 16));
        if ( !(unsigned __int8)sub_2EC9220(*(_DWORD *)(a3 + 40), *(_DWORD *)(a2 + 40), a3, a2, 9u)
          && !(unsigned __int8)sub_2EC9250(*(_DWORD *)(a3 + 44), *(_DWORD *)(a2 + 44), a3, a2, 0xAu)
          && (*(_BYTE *)(a1 + 36) || !*(_BYTE *)a3 || *(_BYTE *)(a1 + 52) || !(unsigned __int8)sub_2EC9280(a3, a2, a4)) )
        {
          v22 = *(_DWORD *)(*(_QWORD *)(a3 + 16) + 200LL);
          v23 = *(_DWORD *)(*(_QWORD *)(a2 + 16) + 200LL);
          if ( a4[6] == 1 )
          {
            if ( v22 >= v23 )
              return v9;
          }
          else if ( v22 <= v23 )
          {
            return v9;
          }
          *(_BYTE *)(a3 + 24) = 15;
          return 1;
        }
        goto LABEL_18;
      }
      if ( !(unsigned __int8)sub_2EC98E0(
                               (_WORD *)(a3 + 34),
                               (__int16 *)(a2 + 34),
                               a3,
                               a2,
                               8u,
                               *(_QWORD *)(a1 + 24),
                               *(_QWORD *)(v18 + 32)) )
      {
        v18 = *(_QWORD *)(a1 + 136);
        goto LABEL_17;
      }
    }
LABEL_18:
    LOBYTE(v9) = *(_BYTE *)(a3 + 24) != 0;
    return v9;
  }
  v21 = *(_QWORD *)(a1 + 136);
  if ( *(_BYTE *)(v21 + 4016)
    && (unsigned __int8)sub_2EC98E0(
                          (_WORD *)(a3 + 34),
                          (__int16 *)(a2 + 34),
                          a3,
                          a2,
                          8u,
                          *(_QWORD *)(a1 + 24),
                          *(_QWORD *)(v21 + 32)) )
  {
    goto LABEL_18;
  }
  return v9;
}
