// Function: sub_A45140
// Address: 0xa45140
//
__int64 __fastcall sub_A45140(__int64 a1, char **a2)
{
  __int64 result; // rax
  unsigned __int8 *v5; // rdx
  __int64 v6; // rsi
  int v7; // eax
  __int64 v8; // rdi
  int v9; // ecx
  int v10; // r8d
  unsigned __int8 *v11; // r14
  unsigned __int8 *v12; // rbx
  char *v13; // rax

  result = sub_A44BF0(a1, a2[1]);
  v6 = *(unsigned __int8 *)a2;
  if ( (unsigned __int8)v6 <= 0x15u )
  {
    v7 = *(_DWORD *)(a1 + 104);
    v8 = *(_QWORD *)(a1 + 88);
    if ( v7 )
    {
      v9 = v7 - 1;
      result = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v5 = *(unsigned __int8 **)(v8 + 16 * result);
      if ( a2 == (char **)v5 )
        return result;
      v10 = 1;
      while ( v5 != (unsigned __int8 *)-4096LL )
      {
        result = v9 & (unsigned int)(v10 + result);
        v5 = *(unsigned __int8 **)(v8 + 16LL * (unsigned int)result);
        if ( a2 == (char **)v5 )
          return result;
        ++v10;
      }
    }
    result = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)a2 + 7) & 0x40) != 0 )
    {
      v12 = (unsigned __int8 *)*(a2 - 1);
      v11 = &v12[result];
    }
    else
    {
      v11 = (unsigned __int8 *)a2;
      v12 = (unsigned __int8 *)a2 - result;
    }
    if ( v11 != v12 )
    {
      do
      {
        if ( **(_BYTE **)v12 != 23 )
          result = sub_A45140(a1);
        v12 += 32;
      }
      while ( v11 != v12 );
      v6 = *(unsigned __int8 *)a2;
    }
    if ( (_BYTE)v6 == 5 )
    {
      result = *((unsigned __int16 *)a2 + 1);
      if ( (_WORD)result == 63 )
      {
        v6 = sub_AC3600(a2);
        sub_A45140(a1);
        result = *((unsigned __int16 *)a2 + 1);
      }
      if ( (_WORD)result == 34 )
      {
        v13 = (char *)sub_BB5290(a2, v6, v5);
        return sub_A44BF0(a1, v13);
      }
    }
  }
  return result;
}
