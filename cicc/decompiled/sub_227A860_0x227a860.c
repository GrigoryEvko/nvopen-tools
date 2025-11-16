// Function: sub_227A860
// Address: 0x227a860
//
__int64 __fastcall sub_227A860(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  int v4; // eax
  int v5; // r8d
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // r9
  __int64 v9; // r8
  _QWORD *v10; // rcx
  __int64 result; // rax
  int v12; // eax
  int v13; // r10d

  v3 = *(_QWORD *)(*(_QWORD *)a1 + 312LL);
  v4 = *(_DWORD *)(*(_QWORD *)a1 + 328LL);
  if ( !v4 )
    goto LABEL_10;
  v5 = v4 - 1;
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v3 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v12 = 1;
    while ( v8 != -4096 )
    {
      v13 = v12 + 1;
      v6 = v5 & (v12 + v6);
      v7 = (__int64 *)(v3 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v12 = v13;
    }
LABEL_10:
    BUG();
  }
LABEL_3:
  v9 = **(_QWORD **)(a1 + 8);
  v10 = (_QWORD *)v7[1];
  result = 0;
  if ( v9 != *v10 )
  {
    sub_D23FE0(v9, *(_QWORD *)(a1 + 16), a2);
    return 1;
  }
  return result;
}
