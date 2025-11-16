// Function: sub_1594A60
// Address: 0x1594a60
//
__int64 __fastcall sub_1594A60(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // edx
  __int64 v6; // rsi
  int v7; // r8d
  __int64 *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // r13

  v1 = sub_16498A0(a1);
  v2 = *(_QWORD *)v1;
  result = *(unsigned int *)(*(_QWORD *)v1 + 1544LL);
  if ( (_DWORD)result )
  {
    v4 = *a1;
    v5 = result - 1;
    v6 = *(_QWORD *)(v2 + 1528);
    v7 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)*a1 >> 9) ^ ((unsigned int)*a1 >> 4));
    v8 = (__int64 *)(v6 + 16 * result);
    v9 = *v8;
    if ( v4 == *v8 )
    {
LABEL_3:
      v10 = v8[1];
      if ( v10 )
      {
        sub_164BE60(v8[1]);
        result = sub_1648B90(v10);
      }
      *v8 = -16;
      --*(_DWORD *)(v2 + 1536);
      ++*(_DWORD *)(v2 + 1540);
    }
    else
    {
      while ( v9 != -8 )
      {
        result = v5 & (unsigned int)(v7 + result);
        v8 = (__int64 *)(v6 + 16LL * (unsigned int)result);
        v9 = *v8;
        if ( v4 == *v8 )
          goto LABEL_3;
        ++v7;
      }
    }
  }
  return result;
}
