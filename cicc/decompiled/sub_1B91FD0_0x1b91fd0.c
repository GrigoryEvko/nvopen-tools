// Function: sub_1B91FD0
// Address: 0x1b91fd0
//
__int64 __fastcall sub_1B91FD0(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r12d
  int v5; // eax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *i; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v17; // rsi
  __int64 v18; // rdi
  unsigned int v19; // ebx

  v3 = sub_1BF29F0(*(_QWORD *)(a1 + 320), *(_QWORD *)(a2 + 40));
  if ( !(_BYTE)v3 )
    return 0;
  v4 = v3;
  v5 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned int)(v5 - 24) <= 0x15 )
  {
    if ( (unsigned int)(v5 - 24) <= 0x13 && (unsigned int)(v5 - 41) > 1 )
      return 0;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v17 = *(_QWORD *)(a2 - 8);
    else
      v17 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v18 = *(_QWORD *)(v17 + 24);
    if ( *(_BYTE *)(v18 + 16) == 13 )
    {
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 <= 0x40 )
      {
        if ( !*(_QWORD *)(v18 + 24) )
          return v4;
      }
      else if ( v19 == (unsigned int)sub_16A57B0(v18 + 24) )
      {
        return v4;
      }
      return 0;
    }
  }
  else
  {
    v4 = 0;
    if ( (unsigned int)(v5 - 54) <= 1 )
    {
      v6 = *(_QWORD *)(a1 + 320);
      v7 = *(_QWORD *)(v6 + 504);
      if ( v7 == *(_QWORD *)(v6 + 496) )
        v8 = *(unsigned int *)(v6 + 516);
      else
        v8 = *(unsigned int *)(v6 + 512);
      v9 = (_QWORD *)(v7 + 8 * v8);
      v10 = sub_15CC2D0(v6 + 488, a2);
      v11 = *(_QWORD *)(v6 + 504);
      if ( v11 == *(_QWORD *)(v6 + 496) )
        v12 = *(unsigned int *)(v6 + 516);
      else
        v12 = *(unsigned int *)(v6 + 512);
      for ( i = (_QWORD *)(v11 + 8 * v12); i != v10; ++v10 )
      {
        if ( *v10 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      v4 = 0;
      if ( v10 != v9 )
      {
        v14 = sub_13A4950(a2);
        v15 = *(_QWORD *)(a1 + 320);
        if ( *(_BYTE *)(a2 + 16) == 54 )
        {
          if ( !(unsigned int)sub_1BF20B0(v15, v14) || !(unsigned __int8)sub_14A2B60(*(_QWORD *)(a1 + 328)) )
            return (unsigned int)sub_14A2B90(*(_QWORD *)(a1 + 328)) ^ 1;
        }
        else if ( !(unsigned int)sub_1BF20B0(v15, v14) || !(unsigned __int8)sub_14A2B30(*(_QWORD *)(a1 + 328)) )
        {
          return (unsigned int)sub_14A2BC0(*(_QWORD *)(a1 + 328)) ^ 1;
        }
        return 0;
      }
    }
  }
  return v4;
}
