// Function: sub_2E225E0
// Address: 0x2e225e0
//
__int64 __fastcall sub_2E225E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 *v9; // rbx
  __int64 *v10; // r12
  __int64 v11; // rsi
  __int64 result; // rax
  unsigned __int64 v13; // rdi
  int v14; // ecx
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rdx

  v8 = *(_QWORD *)(a2 + 32);
  sub_2E221F0(a1, v8, a3, a4, a5, a6);
  v9 = *(__int64 **)(a2 + 112);
  v10 = &v9[*(unsigned int *)(a2 + 120)];
  while ( v10 != v9 )
  {
    v11 = *v9++;
    sub_2E21B90(a1, v11);
  }
  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v13 = result;
  if ( result != a2 + 48 )
  {
    if ( !result )
      BUG();
    v14 = *(_DWORD *)(result + 44);
    v15 = *(_QWORD *)result;
    v16 = *(_DWORD *)(v13 + 44) & 0xFFFFFF;
    if ( (v15 & 4) != 0 )
    {
      if ( (v14 & 4) != 0 )
        goto LABEL_7;
    }
    else if ( (v14 & 4) != 0 )
    {
      while ( 1 )
      {
        v13 = v15 & 0xFFFFFFFFFFFFFFF8LL;
        LOBYTE(v16) = *(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 44);
        if ( (v16 & 4) == 0 )
          break;
        v15 = *(_QWORD *)v13;
      }
    }
    if ( (v16 & 8) != 0 )
    {
      result = sub_2E88A90(v13, 32, 1);
LABEL_8:
      if ( (_BYTE)result )
      {
        v17 = *(_QWORD *)(v8 + 48);
        if ( *(_BYTE *)(v17 + 120) )
          return sub_2E21C60(a1, *(_QWORD *)(v8 + 32), v17);
      }
      return result;
    }
LABEL_7:
    result = (*(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL) >> 5) & 1LL;
    goto LABEL_8;
  }
  return result;
}
