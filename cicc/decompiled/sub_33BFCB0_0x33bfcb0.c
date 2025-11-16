// Function: sub_33BFCB0
// Address: 0x33bfcb0
//
void __fastcall sub_33BFCB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rax
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  int v9; // edx
  int v10; // r9d

  if ( !(unsigned __int8)sub_BCADB0(*(_QWORD *)(a2 + 8)) )
  {
    v3 = *(_QWORD *)(a1 + 960);
    v4 = *(_QWORD *)(v3 + 128);
    v5 = *(unsigned int *)(v3 + 144);
    if ( (_DWORD)v5 )
    {
      v6 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
      {
LABEL_5:
        if ( v7 != (__int64 *)(v4 + 16 * v5) )
          sub_33BF9C0(a1, a2, *((_DWORD *)v7 + 2), 215);
      }
      else
      {
        v9 = 1;
        while ( v8 != -4096 )
        {
          v10 = v9 + 1;
          v6 = (v5 - 1) & (v9 + v6);
          v7 = (__int64 *)(v4 + 16LL * v6);
          v8 = *v7;
          if ( a2 == *v7 )
            goto LABEL_5;
          v9 = v10;
        }
      }
    }
  }
}
