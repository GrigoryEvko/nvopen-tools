// Function: sub_2E39F50
// Address: 0x2e39f50
//
__int64 *__fastcall sub_2E39F50(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rcx
  int v4; // eax
  int v5; // r9d
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r10
  int v9; // eax
  __int64 v10; // rsi
  __int64 *result; // rax
  __int64 v12; // rdx
  int v13; // eax
  int v14; // r11d
  __int64 *v15; // [rsp+10h] [rbp-20h] BYREF
  __int64 v16; // [rsp+18h] [rbp-18h]

  v2 = *a1;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 168);
    v4 = *(_DWORD *)(v2 + 184);
    if ( v4 )
    {
      v5 = v4 - 1;
      v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = (__int64 *)(v3 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
      {
LABEL_4:
        v9 = *((_DWORD *)v7 + 2);
LABEL_5:
        v10 = **(_QWORD **)(v2 + 128);
        LODWORD(v15) = v9;
        result = sub_FE8990(v2, v10, (unsigned int *)&v15, 0);
        v15 = result;
        v16 = v12;
        return result;
      }
      v13 = 1;
      while ( v8 != -4096 )
      {
        v14 = v13 + 1;
        v6 = v5 & (v13 + v6);
        v7 = (__int64 *)(v3 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_4;
        v13 = v14;
      }
    }
    v9 = -1;
    goto LABEL_5;
  }
  LOBYTE(v16) = 0;
  return v15;
}
