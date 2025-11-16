// Function: sub_B98000
// Address: 0xb98000
//
__int64 __fastcall sub_B98000(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 *v8; // rbx
  __int64 v9; // rcx
  int v10; // r8d
  unsigned __int8 v11; // [rsp-29h] [rbp-29h]

  result = 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v3 = a2;
    v4 = *(_QWORD *)sub_BD5C60(a1, a2);
    v5 = *(unsigned int *)(v4 + 3248);
    v6 = *(_QWORD *)(v4 + 3232);
    if ( (_DWORD)v5 )
    {
      v7 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v8 = (__int64 *)(v6 + 40LL * v7);
      v9 = *v8;
      if ( a1 == *v8 )
        goto LABEL_4;
      v10 = 1;
      while ( v9 != -4096 )
      {
        v7 = (v5 - 1) & (v10 + v7);
        v8 = (__int64 *)(v6 + 40LL * v7);
        v9 = *v8;
        if ( a1 == *v8 )
          goto LABEL_4;
        ++v10;
      }
    }
    v8 = (__int64 *)(v6 + 40 * v5);
LABEL_4:
    result = sub_B97D20(v8 + 1, v3);
    if ( !*((_DWORD *)v8 + 4) )
    {
      v11 = result;
      sub_B91E30(a1, v3);
      return v11;
    }
  }
  return result;
}
