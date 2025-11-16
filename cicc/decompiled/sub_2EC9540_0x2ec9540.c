// Function: sub_2EC9540
// Address: 0x2ec9540
//
__int64 __fastcall sub_2EC9540(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  _DWORD *v6; // r14
  unsigned int v7; // r15d
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  char v14; // al

  v5 = *(_QWORD *)(a2 + 16);
  if ( v5 )
  {
    v6 = (_DWORD *)(a1 + 144);
    v7 = sub_2EC8BB0(a1 + 144, v5);
    v8 = sub_2EC8BB0(a1 + 144, *(_QWORD *)(a3 + 16));
    if ( (unsigned __int8)sub_2EC9220(v8, v7, a3, a2, 5u) )
      goto LABEL_9;
    v9 = *(_QWORD *)(a1 + 136);
    v10 = *(_QWORD *)(v9 + 3520);
    v11 = *(_QWORD *)(v9 + 3528);
    v12 = v10;
    if ( *(_BYTE *)(a2 + 25) )
      v12 = v11;
    if ( !*(_BYTE *)(a3 + 25) )
      v11 = v10;
    if ( (unsigned __int8)sub_2EC9250(*(_QWORD *)(a3 + 16) == v11, *(_QWORD *)(a2 + 16) == v12, a3, a2, 6u) )
      goto LABEL_9;
    if ( (unsigned __int8)sub_2EC9220(*(_DWORD *)(a3 + 40), *(_DWORD *)(a2 + 40), a3, a2, 9u) )
      goto LABEL_9;
    v7 = sub_2EC9250(*(_DWORD *)(a3 + 44), *(_DWORD *)(a2 + 44), a3, a2, 0xAu);
    if ( (_BYTE)v7 )
      goto LABEL_9;
    v14 = *(_BYTE *)(a2 + 25);
    if ( v14 != *(_BYTE *)(a3 + 25) || !*(_BYTE *)a2 )
      goto LABEL_14;
    if ( !v14 )
      v6 = (_DWORD *)(a1 + 864);
    if ( (unsigned __int8)sub_2EC9280(a3, a2, v6) )
    {
LABEL_9:
      LOBYTE(v7) = *(_BYTE *)(a3 + 24) != 0;
    }
    else
    {
LABEL_14:
      if ( *(_DWORD *)(*(_QWORD *)(a3 + 16) + 200LL) < *(_DWORD *)(*(_QWORD *)(a2 + 16) + 200LL) )
      {
        *(_BYTE *)(a3 + 24) = 15;
        return 1;
      }
    }
  }
  else
  {
    *(_BYTE *)(a3 + 24) = 16;
    return 1;
  }
  return v7;
}
