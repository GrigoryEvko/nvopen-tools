// Function: sub_2BDF830
// Address: 0x2bdf830
//
unsigned __int8 *__fastcall sub_2BDF830(__int64 a1)
{
  unsigned __int8 *result; // rax
  unsigned __int8 *v2; // rcx
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rax
  size_t v7; // r12
  unsigned __int8 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rdx
  unsigned __int8 v12; // [rsp+Fh] [rbp-31h]

  result = *(unsigned __int8 **)(a1 + 176);
  v2 = *(unsigned __int8 **)(a1 + 184);
  if ( result == v2 )
    goto LABEL_22;
  *(_QWORD *)(a1 + 176) = result + 1;
  v4 = *(_QWORD *)(a1 + 192);
  v5 = *result;
  if ( (*(_BYTE *)(*(_QWORD *)(v4 + 48) + 2 * v5 + 1) & 8) == 0 )
  {
    if ( (_BYTE)v5 == 44 )
    {
      *(_DWORD *)(a1 + 144) = 25;
      return result;
    }
    if ( (*(_DWORD *)(a1 + 140) & 0x120) != 0 )
    {
      if ( (_BYTE)v5 == 92 && v2 != result + 1 && result[1] == 125 )
      {
        *(_DWORD *)(a1 + 136) = 0;
        result += 2;
        *(_DWORD *)(a1 + 144) = 13;
        *(_QWORD *)(a1 + 176) = result;
        return result;
      }
    }
    else if ( (_BYTE)v5 == 125 )
    {
      *(_DWORD *)(a1 + 136) = 0;
      *(_DWORD *)(a1 + 144) = 13;
      return result;
    }
LABEL_22:
    abort();
  }
  *(_DWORD *)(a1 + 144) = 26;
  sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, *(_QWORD *)(a1 + 208), 1u, v5);
  for ( result = *(unsigned __int8 **)(a1 + 176);
        result != *(unsigned __int8 **)(a1 + 184);
        result = *(unsigned __int8 **)(a1 + 176) )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 48LL) + 2LL * *result + 1) & 8) == 0 )
      break;
    v7 = *(_QWORD *)(a1 + 208);
    *(_QWORD *)(a1 + 176) = result + 1;
    v8 = *result;
    v9 = *(_QWORD *)(a1 + 200);
    v10 = v7 + 1;
    v11 = v9 == a1 + 216 ? 15LL : *(_QWORD *)(a1 + 216);
    if ( v10 > v11 )
    {
      v12 = v8;
      sub_2240BB0((unsigned __int64 *)(a1 + 200), v7, 0, 0, 1u);
      v9 = *(_QWORD *)(a1 + 200);
      v8 = v12;
    }
    *(_BYTE *)(v9 + v7) = v8;
    v6 = *(_QWORD *)(a1 + 200);
    *(_QWORD *)(a1 + 208) = v10;
    *(_BYTE *)(v6 + v7 + 1) = 0;
  }
  return result;
}
